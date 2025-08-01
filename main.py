"""
Traffic Technique - Adaptive Control Module

Authors: Georgios Terzoglou, Manos Nestwras  
Organization: Traffic Technique  
Date: 2024-12-29  

Description:  
This is the main entry point for the Adaptive Network Control system.  
It initializes configurations, handles data flow between Firebase, MongoDB, and SUMO,  
and manages the main control loop for adaptive traffic management.  

Run the script with:  
    python main.py  

Dependencies:  
- Python 3.10.8  
- Required libraries:  
    - pymongo  
    - firebase-admin  
    - tensorflow==2.12.0  
    - scikit-learn  
    - pandas  
    - python-dotenv==0.19.2  
    - flask_pymongo  
    - openpyxl  
"""


from utils import * 
from firebase import FirebaseHandler  
from mongo import MongoDBHandler  
from sumo import SUMOHandler 
import gc 
import time  
from flask_pymongo import pymongo  
import socket
import subprocess

from routeSampler import runRouteSampler  
from routeSampler import get_options as RS_get_options 

from custom_sumo import runSUMO
import sys
from transformersHandler import TransformerTimeSeriesHandler

logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
sys.excepthook = log_exception


def main(env_config):
    print('Traffic Technique - Adaptive Control Module')

    # Load configurations and initialize handlers
    firebase_handler = FirebaseHandler(env_config['paths'])
    mongodb_handler = MongoDBHandler(env_config['paths']['mongo_url'])
    firebase_config = firebase_handler.retrieve_data("configuration/" + env_config["project"])

    # Validate license
    # if not licence(firebase_config, env_config["token"], env_config["encryption_key"]):
    #     print("License error!!! Please contact Traffic Technique.")
    #     error_message = "License error!!!"
    #     logging.error("🚨 Error in traffic control loop:\n%s", error_message)
    #     send_email(env_config, "🚨 Adaptive System Crash Alert", f"An error occurred:\n\n{error_message}")
    #     exit()

    # Configure SUMO paths and initialize variables
    configure_sumo_paths(env_config)
    tcID, detlist = load_intersection_ids(firebase_config)
    error_data = initialize_error_lists(tcID, detlist)
    global_vars = initialize_global_variables(firebase_config)

    # Create mean week for each tcid
    current_timestamp = datetime.now()
    mean_weeks = []

    for i in range(len(tcID)):
        temp = mongodb_handler.get_5min_traffic_data(tcID[i], current_timestamp, num_values=25920)    # three months 25920, one week 2016
        mean_df = compute_mean_week(temp, tc_id=tcID[i])
        mean_weeks.append(mean_df)
    dataset, nan_flag, peak_periods, threshold_stats = create_initial_traffic_dataset(tcID, mongodb_handler, mean_weeks, global_vars['timeStamp'])
    del temp
    gc.collect()
    sumo_handler = SUMOHandler(firebase_config, env_config, peak_periods=peak_periods, threshold_stats=threshold_stats)

    mongodb_handler.disconnect()

    # Create handlers to make predictions using transformers
    handlers = {}

    for i, df in enumerate(dataset):
        for col in df.columns:
            name = f"tcID{tcID[i]}{col}"
            handler = TransformerTimeSeriesHandler(model_dir=env_config["paths"]["trans_modelsPATH"], name=name)
            handlers[name] = handler

    # Time interval in minutes
    sMin = 5

    training_proc = None

    detectors = [["H" + str(i + 1) for i in range(n)] for n in detlist]
    sumo_data = {"sumo_flows": [], "sumo_turns": []}
    
    while True:
        try:
            # Refresh authentication token
            global_vars['stc_token'], global_vars['tokentimestamp'] = handle_token_refresh(
                global_vars['stc_token'], global_vars['tokentimestamp'],
                env_config["email"], env_config["password"], env_config["stcweb_url"]
            )

            # Calculate remaining time to next run
            wait_time = get_seconds_until_next_run(sMin, firebase_config)
            print(f"Next run in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

            current_timestamp = datetime.now()
            print(f"Processing traffic data at {current_timestamp.strftime('%H:%M:%S')}")
            # Execute code

            # Connect to MongoDB and fetch traffic data
            dbClient = pymongo.MongoClient(env_config['paths']['mongo_url'])
            trafficDatadb = dbClient["his_1234"]
            col = trafficDatadb["traffic_data"]
            reinitialize_global_variables(global_vars)

            # Calculate lost time on green start
            lstgr = [sum(firebase_config[str(tcID[i])]["stages"][j]["lost_green"]
                            for j in firebase_config[str(tcID[i])]["stages"]) for i in range(len(tcID))]

            # Process traffic data and predictions
            data_arr, no_act, pred_dis = process_traffic_data(
                tcID, sMin, current_timestamp, col, firebase_config, mean_weeks
            )

            eval_data_arr = [[0 for i in range(len(n))] for n in data_arr]

            if no_act == 0:
                # Update datasets and handle NaN values
                data_arr, nan_flag, dataset = replace_nan_values(
                        data_arr, nan_flag, dataset, tcID, mean_weeks, current_timestamp
                    )

                if pred_dis == 0:
                    # Perform predictions and log results
                    predData = perform_predictions_transformers(
                        dataset=dataset,
                        tcID=tcID,
                        firebase_config=firebase_config,
                        handlers=handlers
                    )
                    log_data_to_firebase(
                        current_timestamp, data_arr, predData, firebase_config, tcID, env_config["project"]
                    )
                else:
                    predData = np.full((len(tcID), len(dataset[0].columns)), np.nan)

                if global_vars["eval_run"]:
                    eval_data_arr = calculate_evaluation_data(
                        data_arr, eval_data_arr, firebase_config, global_vars['queued_arr']
                    )
                # Calculate queues
                global_vars['queued_arr'] = calculate_queues(
                    tcID, data_arr, global_vars["eval_dsteps_arr"], global_vars["eval_c_index"], sMin, firebase_config
                )

                # Process errors and adapt traffic data
                error_data = process_prediction_errors(
                    firebase_handler, pred_dis, tcID, detectors, error_data, current_timestamp, env_config["project"]
                )
            
                data_arr = adapt_and_update_traffic_data(
                    tcID, data_arr, predData, global_vars['queued_arr'], error_data, firebase_config, global_vars["eval_run"]
                )

                # Compute SUMO flows and traffic conditions
                global_vars["tr_stg"], global_vars["global_traffic"], sumo_data = sumo_handler.compute_sumo_and_algo_flows(
                    tcID, data_arr, sumo_data, global_vars["tr_stg"], global_vars["global_traffic"]
                )
            # Handle traffic scenarios
            c_max_mode, highTraffic, no_act = check_traffic_conditions(global_vars["global_traffic"], firebase_config)
            if no_act == 0:
                sumo_handler.flowtoxml(sumo_data["sumo_flows"])
                sumo_handler.turnstoxml(sumo_data["sumo_turns"])
                runRouteSampler(RS_get_options(cmdl=True))

                for cycle in firebase_config["cycles"]:
                    steps_arr = sumo_handler.calculate_decision_steps(
                        tcID, cycle_length=firebase_config["cycles"][cycle],
                        tr_stg=global_vars["tr_stg"], global_traffic=global_vars["global_traffic"], lstgr=lstgr
                    )
                    offsets_c_arr, offsetsSumo_c_arr = sumo_handler.calculate_offsets(
                        tcID, cycle_length=firebase_config["cycles"][cycle], decision_steps=steps_arr
                    )
                    global_vars["dsteps_arr"].append(steps_arr)
                    global_vars["offsets_arr"].append(offsets_c_arr)
                    global_vars["offsetsSumo_arr"].append(offsetsSumo_c_arr)

                    # Run SUMO simulations and evaluate
                    sumoStat = runSUMO()
                    global_vars["tryCycles"].append(firebase_config["cycles"][cycle])
                    global_vars["time_loss"].append(float(sumoStat[0]))

                select_best_cycle(global_vars, highTraffic)

                select_optimal_cycle(global_vars, c_max_mode)

                c_index = global_vars["tryCycles"].index(global_vars["previous_best_cycle"])
                process_offsets_and_decision_steps(
                    sumo_handler, tcID, firebase_config, global_vars["offsetsSumo_arr"], 
                    global_vars["dsteps_arr"], c_index
                )
                
                process_algo_command_and_logs(
                    tcID, firebase_config, global_vars, c_index, env_config, current_timestamp
                )
            # Handle low traffic conditions
            else:
                global_vars = handle_low_traffic_condition(global_vars, tcID, firebase_config, env_config, current_timestamp)

            # Select fixed time plan for junctions
            select_fixed_time_plan_junctions(
                firebase_config, global_vars["stc_token"], global_vars["previous_best_cycle"], env_config["stcweb_url"]
            )

            if no_act == 0:
                # Evaluate SUMO run
                evaluate_sumo_run(
                    no_act, firebase_config, global_vars, tcID, eval_data_arr, sumo_handler, 
                    firebase_handler, current_timestamp, env_config, sumo_data, c_index,
                    peak_periods, threshold_stats
                )
            dbClient.close()
            sumo_data["sumo_flows"] = []
            sumo_data["sumo_turns"] = []
            del dbClient, trafficDatadb, col, data_arr, pred_dis
            gc.collect()

            if should_retrain_now(global_vars["retrain_time"]):
                if training_proc is not None and training_proc.poll() is None:
                    # Trainer is still running – skip
                    pass
                else:
                    print("Retraining triggered: Recomputing mean_week and peak_periods...")

                    # Recalculate mean_week
                    mean_weeks = []
                    for i in range(len(tcID)):
                        temp = mongodb_handler.get_5min_traffic_data(tcID[i], current_timestamp, num_values=25920)
                        mean_df = compute_mean_week(temp, tc_id=tcID[i])
                        mean_weeks.append(mean_df)
                    del temp
                    gc.collect()

                    # Recalculate peak_periods and threshold_stats
                    dataset, nan_flag, peak_periods, threshold_stats = create_initial_traffic_dataset(
                        tcID, mongodb_handler, mean_weeks, global_vars['timeStamp']
                    )

                    # Launch the training script
                    training_proc = subprocess.Popen(["python", "train_transformers.py"])

            # Reload Firebase configuration if necessary
            firebase_config = firebase_handler.retrieve_data("configuration/" + env_config["project"])
            if firebase_config["restart"] == 1:
                firebase_handler.restart_config(env_config["project"])
                print("System restart triggered.")
                exit()
            
            # 🔄 Check for update request
            if firebase_config.get("update") == 1:
                print("🔄 Update requested from Firebase...")
 
                # Reset update flag in Firebase to avoid loops
                firebase_handler.update_config(env_config["project"])
 
                try:
                    subprocess.run(["git", "fetch", "origin"], check=True)
                    # Discard any local changes
                    subprocess.run(["git", "reset", "--hard"], check=True)
                    #subprocess.run(["git", "clean", "-fd"], check=True)  # remove untracked files (like .pyc)
                   
                    #subprocess.run(["git", "checkout", "main"], check=True)
                    subprocess.run(["git", "pull", "origin", "main"], check=True)
                    print("✅ Code updated to latest main branch.")
                except subprocess.CalledProcessError as e:
                    print(f"⚠ Git update failed: {e}")
                    logging.error("Git update failed:\n%s", traceback.format_exc())
                    continue
                print("♻ Restarting main.py with updated code...")
                sys.exit(0)
                
        except Exception as e:
            error_message = traceback.format_exc()
            logging.error("🚨 Error in traffic control loop:\n%s", error_message)
            send_email(env_config, "🚨 Adaptive System Crash Alert", f"An error occurred:\n\n{error_message}")
            exit()


if __name__ == "__main__":
    env_config = load_config()
    # Create a socket object
    s = socket.socket()           
    # Get local machine name
    host = socket.gethostname()    
    # Reserve a port
    port = env_config['port'] 
    # Bind to the port
    try:
        s.bind((host, port))   
    except socket.error as msg:
        print("Another instance is running. Exit")
        exit()
    main(env_config)
