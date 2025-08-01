"""
Traffic Technique - Adaptive Control Module

Authors: Georgios Terzoglou, Manos Nestwras  
Organization: Traffic Technique  
Date: 2024-12-29  

Description:  
This module provides utility functions for configuration loading,  
global variable initialization, and data preprocessing.  
It supports common tasks used across different modules of the adaptive control system.  
"""


import os
from dotenv import load_dotenv
import math
import numpy as np
import pandas as pd
import uuid
import jwt
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import requests
import json
from bson import json_util
import pickle
from multiprocessing import Pool
import multiprocessing
from firebase_admin import credentials, db
from stcwebConnect import plan_selection, plan_params, stcweb_login
from tensorflow.keras.models import load_model
import math
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import logging
import sys
import traceback
import ssl

from custom_sumo import runSUMO


def load_config():
    """Load environment variables and configurations."""
    current_path = os.getcwd()
    load_dotenv(os.path.join(current_path, '.env'))
    return {
        "project": os.getenv('project'),
        "paths": {
            "filePATH": os.getenv('filePATH'),
            "sumoPATH": os.getenv('sumoPATH'),
            "sumocfg": os.getenv('sumocfg'),
            "netfile": os.getenv('net-file'),
            "routefiles": os.getenv('route-files'),
            "additionalfiles": os.getenv('additional-files'),
            "statisticoutput": os.getenv('statistic-output'),
            "guisettingsfile": os.getenv('gui-settings-file'),
            "modelsPATH": os.getenv('modelsPATH'),
            "trans_modelsPATH": os.getenv('trans_modelsPATH'),
            "mongo_url": os.getenv('mongo_url'),
            "Firebase_url": os.getenv('Firebase_url'),
            "Firebase_key": os.getenv('Firebase_key')
        },
        "stcweb_url": os.getenv('stcweb_url'), 
        "token": os.getenv('valid_token'),
        "email": os.getenv('EMAIL'),
        "password": os.getenv('PASSWORD'),
        "email_sender": os.getenv('EMAIL_SENDER'),
        "email_sender_password": os.getenv('EMAIL_SENDER_PASSWORD'),
        "email_receiver": os.getenv('EMAIL_RECEIVER'),
        "port": int(os.getenv('port')),
        "encryption_key": 'hjK&aW2%1!vf*(0AC690'
    }

def send_email(config, subject, body):
    """ Sends an email notification using COSMOTE Mail """
    try:
        # SMTP Configuration (COSMOTE Mail)
        SMTP_SERVER = "mailgate.cosmotemail.gr"
        SMTP_PORT = 587
        EMAIL_SENDER = config["email_sender"]
        EMAIL_SENDER_PASSWORD = config["email_sender_password"] 
        EMAIL_RECEIVER = config["email_receiver"]
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = config["project"]
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Set up secure TLS context
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2  # Ensure secure connection

        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls(context=context)  # Secure the connection

        # Authenticate and send email
        server.login(EMAIL_SENDER, EMAIL_SENDER_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()

        print(f"Error email sent successfully to {EMAIL_RECEIVER}")

    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication failed. Check your email/password.")
    except smtplib.SMTPConnectError:
        logging.error("Unable to connect to SMTP server. Check network or firewall settings.")
    except smtplib.SMTPException as smtp_error:
        logging.error(f"SMTP error: {smtp_error}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


def log_exception(exc_type, exc_value, exc_traceback):
    env_config = load_config()
    """ Logs unhandled exceptions globally and sends an email """
    # if issubclass(exc_type, KeyboardInterrupt):
    #     sys.__excepthook__(exc_type, exc_value, exc_traceback)
    #     return
    # error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    # logging.error("Unhandled exception:\n%s", error_message)
    # print("⚠️ System Error logged and email sent.")
    # send_email(env_config, "🚨 Adaptive System Crash Alert", f"An error occurred:\n\n{error_message}")


def queue(volume, occupancy, interval, detector_setback, multilane, main_road):
    """
    Calculates the queue length (vehicles) based on input parameters.

    Parameters:
    - volume (int/float): Traffic volume (vehicles per time interval).
    - occupancy (int/float): Occupancy percentage (0-100).
    - interval (int/float): Data aggregation time in minutes.
    - detector_setback (int/float): Distance of the detector from the stop bar in meters.
    - multilane (str): Indicates whether the road is multilane ("True"/"False").
    - main_road (str): Indicates whether the road is a main road ("True"/"False").

    Returns:
    - int: Estimated queue length (number of vehicles).
    """
    # Define free flow and saturated speeds based on road type
    Vf = 20 if main_road == "True" else 14  # Free flow speed (m/s)
    V2 = 5 if main_road == "True" else 4    # Saturated speed (m/s)

    # Queue growth adjustment factor (default set to 1 for all cases)
    if 120 <= detector_setback:
        Fq = 1
    if 100 <= detector_setback < 120:
        Fq = 1
    if 80 <= detector_setback < 100:
        Fq = 1
    if 60 <= detector_setback < 80:
        Fq = 1
    if detector_setback < 60:
        Fq = 1

    # Validation of traffic data
    if (
        occupancy == 255 or occupancy > 100 or 
        volume == 255 or math.isnan(occupancy) or 
        math.isnan(volume) or volume in ["-", ""]
    ):
        return 0  # Invalid data, return no queue
    
    # Handle cases with no significant traffic
    if occupancy <= 5 or volume == 0:
        return 0  # No queue under these conditions

    # Adjust volume for multilane roads
    volume_adjusted = volume * 0.97 if multilane == "True" else volume

    # Normalize occupancy percentage
    occ = occupancy / 100

    # Constants for queue calculations
    Ld = 2        # Detector zone length (meters)
    Lv = 4.5      # Average vehicle length (meters)
    Lq = 5.2      # Average vehicle spacing in a queue (meters)
    aggregation_time = interval * 60  # Convert interval to seconds
    h = aggregation_time / volume_adjusted  # Discharge headway (seconds per vehicle)
    
    # Thresholds for occupancy levels
    Occ1 = ((Ld + Lv) / Vf) / h  # Threshold for no queue
    Occ2 = ((Ld + Lv) / V2) / h  # Threshold for queue exceeding detector

    # Queue length calculation
    if occ > Occ2:  # Queue exceeds detector
        Q = Fq * (occ - Occ2) * aggregation_time / h
    else:
        Q = 0  # No significant queue

    # Return the queue length rounded to the nearest integer
    return round(Q)


def queue_calc(volume, occupancy, interval=5):
    """
    Calculates the estimated queue length based on volume and occupancy data,
    incorporating adjustments for volumes that exceed a detector setback.

    Args:
        volume (float): The volume of vehicles.
        occupancy (float): The percentage of time the detector is occupied (0-100).
        interval (int, optional): The data aggregation time in minutes. Defaults to 5.

    Returns:
        float: The estimated queue length plus volume.
    """
    # Handle invalid cases
    if math.isnan(occupancy) or math.isnan(volume) or volume == "-" or volume > 150 or occupancy > 100:
        return np.nan
    if volume == 0:
        return 0

    # Constants and parameters (specific to your domain)
    vol = volume
    occ = occupancy / 100  # Convert occupancy percentage to a fraction
    Ld = 2  # Detector zone length (meters)
    V2 = 5  # Saturated speed (m/s)
    Lv = 4.5  # Average vehicle length (meters)
    Lq = 5.2  # Average vehicle spacing in queue (meters)
    agrTime = interval * 60  # Interval in seconds
    h = agrTime / vol  # Discharge headway at the detector (seconds per vehicle)
    Fq = 1.1  # Queue growth adjustment factor
    Q = 0  # Initial queue length

    # Calculate occupancy threshold for queues that exceed detector setback
    Occ2 = ((Ld + Lv) / V2) / h

    # If occupancy exceeds threshold, calculate the queue length
    if occ > Occ2:
        Q = Fq * (occ - Occ2) * agrTime / h

    # Return the sum of volume and calculated queue length
    return round(vol + Q)

def get_seconds_until_next_run(sMin, firebase_config):
        now = datetime.now()

        # Round at next 5 min
        next_minute = (now.minute // sMin + 1) * sMin
        
        # Create next run datetime
        next_run = now.replace(second=firebase_config["loopOffset"], microsecond=0) + timedelta(minutes=(next_minute - now.minute))        

        return (next_run - now).total_seconds()

def replace_nans(dataframe, mean_weak_df):
    """Replaces NaN values in a DataFrame using previous values or mean week values."""
    for col in dataframe.columns:
        if col != "TimeStamp":
            prev_val = None
            for i, val in enumerate(dataframe[col]):
                if pd.isna(val):
                    if prev_val is None:
                        try:
                            dataframe.at[i, col] = round(value_from_mean_week(mean_weak_df, col, dataframe.at[i, "TimeStamp"]))
                        except:
                            dataframe.at[i, col] = 0
                    elif pd.isna(dataframe.at[i+1, col]):
                        try:
                            dataframe.at[i, col] = round(value_from_mean_week(mean_weak_df, col, dataframe.at[i, "TimeStamp"]))
                            prev_val = None
                        except:
                            dataframe.at[i, col] = 0
                            prev_val = None
                    else:
                        dataframe.at[i, col] = prev_val
                else:
                    prev_val = val
    return dataframe

def value_from_mean_week(mean_week_df, col_name, timestamp):
    """
    Retrieves a value from a mean week DataFrame based on the given timestamp.

    Parameters:
    - mean_week_df (pd.DataFrame): DataFrame containing weekly aggregated data with columns for 'day', 'hour', and 'minute'.
    - col_name (str): The column name from which the value is to be retrieved.
    - timestamp (datetime): A datetime object representing the target timestamp.

    Returns:
    - The value from the specified column corresponding to the day, hour, and minute of the given timestamp.

    Raises:
    - KeyError: If the specified column does not exist in the DataFrame.
    - IndexError: If no matching row is found for the given day, hour, and minute.
    """
    # Extract day, hour, and minute from the timestamp
    day = timestamp.strftime("%A")  # e.g., "Monday"
    hour = timestamp.hour           # Extract the hour (0-23)
    minute = timestamp.minute       # Extract the minute (0-59)
    
    try:
        # Filter the DataFrame for the matching day, hour, and minute
        result = mean_week_df.loc[
            (mean_week_df['day'] == day) &
            (mean_week_df['hour'] == hour) &
            (mean_week_df['minute'] == minute)
        ]
        
        # Return the value from the specified column
        return result[col_name].values[0]
    except KeyError:
        raise KeyError(f"The column '{col_name}' does not exist in the DataFrame.")
    except IndexError:
        raise IndexError(
            f"No matching data found for day='{day}', hour={hour}, minute={minute}."
        )

# License check
def licence(config, token, encryption_key):
    
    # Decode the token using JWT
    decoded_data = jwt.decode(token, encryption_key, algorithms=["HS256"])
    
    # Extract and process license information
    valid_tc_ids = [int(tc_id) for tc_id in decoded_data["Licence"].split(",")]
    valid_mac_address = decoded_data["macAdress"]
    current_mac_address = hex(uuid.getnode()).lower().replace('0x', '')

    # Get adtcID values from config
    param_tc_ids = [config["adtcID"][i] for i in config["adtcID"]]

    # Validate IDs and MAC address
    return valid_tc_ids == param_tc_ids and valid_mac_address == current_mac_address


def configure_sumo_paths(config):
    """Updates SUMO configuration paths in the XML configuration file."""
    filename = os.path.join(config['paths']['sumoPATH'], config['paths']['sumocfg'])
    xmlTree = ET.parse(filename)
    rootElement = xmlTree.getroot()
    for index in rootElement:
        for element in index:
            match element.tag:
                case 'net-file':
                    element.attrib['value'] = config['paths']['netfile']
                case 'route-files':
                    element.attrib['value'] = config['paths']['routefiles']
                case 'additional-files':
                    element.attrib['value'] = config['paths']['additionalfiles']
                case 'statistic-output':
                    element.attrib['value'] = config['paths']['statisticoutput']
                case 'gui-settings-file':
                    element.attrib['value'] = config['paths']['guisettingsfile']
    xmlTree.write(filename, encoding='UTF-8', xml_declaration=True)

def load_intersection_ids(config_data):
    """Loads intersections' IDs and detectors from configuration data."""
    tcID = [0] * len(config_data["adtcID"])
    detlist = []
    for j, i in enumerate(config_data["adtcID"]):
        tcID[j] = config_data["adtcID"][i]
        detlist.append(config_data[str(tcID[j])]["numdet"])
    return tcID, detlist

def initialize_error_lists(tcID, detlist):
    """Initializes error lists and FIFO queues for tracking predictions and previous errors."""
    err_pred = [[0] * n for n in detlist]
    err_prev = [[0] * n for n in detlist]
    fifo_prev_err = [[] for _ in range(len(tcID))]
    fifo_pred_err = [[] for _ in range(len(tcID))]
    for i in range(len(tcID)):
        for _ in range(detlist[i]):
            fifo_prev_err[i].append([])
            fifo_pred_err[i].append([])
    return {
        'err_pred': err_pred,
        'err_prev': err_prev,
        'fifo_prev_err': fifo_prev_err,
        'fifo_pred_err': fifo_pred_err
    }

def initialize_global_variables(config_data):
    """Initializes global variables for evaluation and traffic control."""
    tokentimestamp = datetime.now()
    return {
        'tokentimestamp': tokentimestamp,
        'improvement_percentage': 0.03,
        'stc_token': None,
        'cycle_counter': 3,
        'tr_stg': [],
        'global_traffic': [],
        'queued_arr': [],
        'c_sumo': [],
        'startup_cycle': config_data["startup_c"],
        'previous_best_cycle': config_data["startup_c"],
        'improvement_percentage': 0.03,
        'retrain_time': {"weekday": 2, "hour": 0},
        'eval_run': False,
        'eval_no_act': 0,
        'eval_c_index': 0,
        'timeStamp': datetime.now(),
        'eval_dsteps_arr': [],
        'dsteps_arr': [],
        'offsets_arr': [],
        'offsetsSumo_arr': [],
        'numdets': 0,
        'tryCycles': [],
        'time_loss': []
    }

def reinitialize_global_variables(global_vars):
    """Reinitializes specific global variables to their default values."""
    global_vars['tr_stg'] = []
    global_vars['global_traffic'] = []
    global_vars['SUMOflows'] = []
    global_vars['SUMOturns'] = []
    global_vars['dsteps_arr'] = []
    global_vars['offsets_arr'] = []
    global_vars['c_sumo']: []
    global_vars['offsetsSumo_arr'] = []
    global_vars['tryCycles'] = []
    global_vars['time_loss'] = []
    global_vars['numdets'] = 0

def extract_peak_periods_and_stats(mean_weak_df):
    # Automatically detect sensor columns
    time_columns = ['day', 'hour', 'minute']
    sensor_columns = [col for col in mean_weak_df.columns if col not in time_columns]

    # 1. Calculate stats
    stats = {}
    for h in sensor_columns:
        mean = mean_weak_df[h].mean()
        std = mean_weak_df[h].std()
        threshold = (mean + 2 * std)   # 2 * std is ~95% of meanmax 
        threshold = threshold * 12     # conversion to hourly data
        stats[h] = {'mean': mean, 'std': std, 'threshold': threshold}

    # 2. Helper function to find peak periods
    def get_peak_timezones_filtered(df, sensor, threshold, max_gap_minutes=15, min_duration_minutes=30):
        filtered = df[df[sensor] > threshold/12][['day', 'hour', 'minute']].copy()
        
        if filtered.empty:
            return pd.DataFrame(columns=["Sensor", "Day", "Start Time", "End Time"])
        
        filtered['time'] = pd.to_datetime(
            filtered['hour'].astype(str).str.zfill(2) + ':' + 
            filtered['minute'].astype(str).str.zfill(2), 
            format='%H:%M'
        )

        grouped = filtered.groupby('day')
        peak_periods = []

        for day, group in grouped:
            group = group.sort_values('time').reset_index(drop=True)
            start_time = group.loc[0, 'time']
            end_time = group.loc[0, 'time']
            
            for i in range(1, len(group)):
                current_time = group.loc[i, 'time']
                diff = (current_time - end_time).seconds / 60

                if diff <= max_gap_minutes:
                    end_time = current_time
                else:
                    duration = (end_time - start_time).seconds / 60
                    if duration >= min_duration_minutes:
                        peak_periods.append([sensor, day, start_time.strftime('%H:%M'), end_time.strftime('%H:%M')])
                    start_time = current_time
                    end_time = current_time
            
            duration = (end_time - start_time).seconds / 60
            if duration >= min_duration_minutes:
                peak_periods.append([sensor, day, start_time.strftime('%H:%M'), end_time.strftime('%H:%M')])

        return pd.DataFrame(peak_periods, columns=["Sensor", "Day", "Start Time", "End Time"])

    # 3. Aggregate peak periods
    all_peak_periods = pd.DataFrame()
    for sensor in sensor_columns:
        threshold = stats[sensor]['threshold']
        peaks_df = get_peak_timezones_filtered(mean_weak_df, sensor, threshold)
        all_peak_periods = pd.concat([all_peak_periods, peaks_df], ignore_index=True)

    # 4. Sort by Sensor, Day, and Time
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_peak_periods['Day'] = pd.Categorical(all_peak_periods['Day'], categories=day_order, ordered=True)
    all_peak_periods = all_peak_periods.sort_values(by=['Sensor', 'Day', 'Start Time']).reset_index(drop=True)

    return all_peak_periods, stats

def create_initial_traffic_dataset(tcID, mongodb_handler, mean_weeks, timeStamp):
    """Creates the initial traffic dataset based on retrieved data from MongoDB."""
    dataset = []
    nan_flag = []
    peak_periods = []
    threshold_stats = []
    for i in range(len(tcID)):
        mean_weak_df = mean_weeks[i]
        periods_df, stats_dict = extract_peak_periods_and_stats(mean_weak_df)
        
        temp = mongodb_handler.get_5min_traffic_data(tcID[i], timeStamp, num_values=12)
        temp = replace_nans(temp, mean_weak_df)
        temp = temp.drop("TimeStamp", axis=1)
        temp = temp[sorted(temp.columns, key=lambda x: int(x[1:]))]

        nan_flag.append([False] * len(temp.columns))
        dataset.append(temp.copy())
        peak_periods.append(periods_df)
        threshold_stats.append(stats_dict)
    return dataset, nan_flag, peak_periods, threshold_stats

def login_to_stcweb(email, password, stcweb_url):
    url = stcweb_url + 'login'
    data = {'email': email, 'password': password}
    headers = None
    response = requests.post(url, data = data, headers = headers)
    stc_token = response.json()
        
    return stc_token

def handle_token_refresh(stc_token, tokentimestamp, email, password, stcweb_url):
    """Refreshes the token if needed."""
    if datetime.now() - tokentimestamp > timedelta(hours=12) or stc_token is None:
        log = login_to_stcweb(email, password, stcweb_url)
        if log.get("auth"):
            print("New token retrieved.")
            return log["token"], datetime.now()
        else:
            print("Token retrieval failed. Wrong username or password. Exiting.")
            exit()
    return stc_token, tokentimestamp

def retrieve_traffic_data(mongodb_handler, env_config, global_vars):
    """Fetches and processes traffic data from MongoDB using the provided configuration.

    Args:
        env_config (dict): The environment configuration containing MongoDB paths.
        global_vars (dict): The global variables dictionary for current state management.

    Returns:
        list: Processed traffic data or an empty list if retrieval fails.
    """
    mongodb_handler.connect()

    try:
        traffic_data = []
        collection_name = "traffic_data"
        col = mongodb_handler.db[collection_name]

        # Loop through each Traffic Controller ID (tcID) and retrieve data
        for tc_id in global_vars['tcID']:
            # Adjust timestamp and interval as per the requirement
            timestamp = global_vars['timeStamp']
            interval_minutes = 5  # Default value, can be adjusted based on needs

            # Group traffic counts and handle potential data aggregation logic
            grouped_data = mongodb_handler.group_counts_by_5min(
                tcID=tc_id, 
                timestamp=timestamp, 
                minutes=interval_minutes, 
                collection_name=collection_name
            )

            # If valid data is retrieved, process it further or append
            if not grouped_data.empty:
                traffic_data.append(grouped_data)
            else:
                print(f"No traffic data found for tcID {tc_id} at {timestamp}.")

    except Exception as e:
        print(f"Error retrieving traffic data: {e}")
        traffic_data = []  # Return empty list on failure

    finally:
        mongodb_handler.disconnect()

    return traffic_data

def proccess_volumes(tcID, interval, timeStamp, col, queue_var, mean_weak_df):
        id = int(tcID)
        min = int(interval)
        timeFetch = timeStamp - timedelta(minutes=min)

        year = int(json.dumps(timeFetch.year)[2:])
        month = int(json.dumps(timeFetch.month))
        day = int(json.dumps(timeFetch.day))
        hour = int(json.dumps(timeFetch.hour))
        minutes = int(json.dumps(timeFetch.minute))

        query = [
            {"$match": {
                "tcID": id,
                "Data.Year": {"$eq": year},
                "Data.Month": {"$eq": month},
                "Data.Day": {"$eq": day},
                "Data.Hour": {"$eq": hour},
                "Data.Minutes": {"$gte": minutes, "$lt": minutes+min},
                }
            },
            {"$unwind": "$Data"},
            {"$project": {
                "data": {
                    "$map": {
                    "input": {"$range": [0, "$Data.N_CountOccu"]},
                    "in": {
                        "i": {"$add": ["$$this", 0]},
                        "counters": {"$arrayElemAt": ["$Data.CountOccu", "$$this"]}
                        }
                    }
                    }
                }
            },
            {"$unwind": "$data"},
            {"$group": {
                "_id": "$data.i",
                "volume": {"$sum": "$data.counters.count"},
                "occupancy": {"$avg": "$data.counters.occupancy"},
                }
            },
            {"$sort": {"_id": +1}},
            {"$unset":"_id"}
        ]
        
        try:
            resultQuery = col.aggregate(query)
            r = json_util.dumps(resultQuery)
            sensordata= json.loads(r)

            arr = [0] * len(sensordata) 
            i = 0

            for i in range(len(sensordata)):
                detector = "H"+ str(i+1)
                volume = round(sensordata[i]["volume"])
                occupancy = round(sensordata[i]["occupancy"])
                Q = queue(volume, occupancy, min, queue_var[str(id)][detector]["detectorSetback"], queue_var[str(id)][detector]["multinane"], 
                    queue_var[str(id)][detector]["mainRoad"])
                
                #count integrity check
                if occupancy == 255 or occupancy > 100 or volume == 255 or math.isnan(occupancy) == True or math.isnan(volume) == True or volume == "-" or volume == "" or (volume == 0 and occupancy == 0) or volume*12 > 1500:
                    volume = np.nan

                # proccessed volume
                if math.isnan(volume) == False:
                    volume_procc = volume + Q
                    # excessive queue check and volume replacement from mean week
                    if queue_var[str(id)][detector]["mainRoad"] == "True":
                        meanmax_volume = round(mean_weak_df[detector].max())
                        if 70 <= occupancy < 80 and volume_procc < 0.5 * meanmax_volume:
                            volume_procc = meanmax_volume
                        if 80 <= occupancy and volume_procc < 0.6 * meanmax_volume:
                            volume_procc = 1.1 * meanmax_volume
                else:
                    volume_procc = volume
                            
                arr[i] = volume_procc
                i = i + 1
                    
            del query, resultQuery, r, sensordata

            return arr

        except Exception as e:
            return("error query MongoDB")
        
# Function to process traffic data
def process_traffic_data(tcID, sMin, timeStamp, col, config, mean_weeks):
    """
    Processes traffic data for multiple traffic control IDs (tcID).

    Parameters:
    - tcID (list): List of traffic control IDs to process.
    - sMin (int): Data aggregation interval in minutes.
    - timeStamp (datetime): The timestamp to process data for.
    - col (str): Column name for data processing.
    - config (dict): Configuration dictionary containing settings for each tcID.
    - mean_weeks (array): Array of the mean weeks

    Returns:
    - tuple:
        - data_arr (list): List of processed data for each tcID.
        - no_act (int): Indicator for no activity (1 if no traffic data was found for any tcID, 0 otherwise).
        - pred_dis (int): Indicator for disabled predictions (1 if all detectors are disabled, 0 otherwise).
    """
    # Initialize variables
    data_arr = []  # List to store processed data
    no_act = 0     # Indicator for no traffic activity
    pred_dis = 0   # Counter for disabled predictions
    num_dets = 0   # Total number of detectors

    for i in range(len(tcID)):
        try:
            # Load mean week data for the current traffic control ID
            mean_week_df = mean_weeks[i]

            # Process volumes for the current tcID
            data = proccess_volumes(tcID[i], sMin, timeStamp, col, config, mean_week_df)

            # Delete the DataFrame to free memory
            del mean_week_df

            # Handle errors or successful processing
            if data == "error":
                print("MongoDB error!!")
                data = []
            if data:
                print(f"tcID {tcID[i]} {data} Processed volumes")
                data_arr.append(data)
            else:
                no_act = 1
                print(f"tcID {tcID[i]} no traffic data")

            # Update detector count
            num_dets += int(config[str(tcID[i])]["numdet"])

            # Check prediction status for each detector
            for j in range(int(config[str(tcID[i])]["numdet"])):
                if config[str(tcID[i])][f"H{j + 1}"]["predict"] == "False":
                    pred_dis += 1
        except Exception as e:
            print(f"Error processing tcID {tcID[i]}: {e}")
            no_act = 1  # Assume no activity if an error occurs

    # Determine if all predictions are disabled
    pred_dis = 1 if num_dets == pred_dis else 0

    return data_arr, no_act, pred_dis

def replace_nan_values(data_arr, nan_flag, dataset, tcID, mean_weeks, timeStamp):
    for i in range(len(tcID)):
        dataset[i] = dataset[i].drop(index=0).reset_index(drop=True)
        mean_weak_df = mean_weeks[i]
        last_row = dataset[i].iloc[-1]

        for v in range(len(data_arr[i])):
            try:
                if math.isnan(float(data_arr[i][v])):
                    if nan_flag[i][v] == False:
                        data_arr[i][v] = int(last_row[v])
                        nan_flag[i][v] = True
                    else:
                        try:
                            data_arr[i][v] = int(round(value_from_mean_week(mean_weak_df, "H"+str(v+1), timeStamp)))
                        except:
                            data_arr[i][v] = int(0)
                else:
                    nan_flag[i][v] = False
            except (ValueError, TypeError):
                # If value cannot be converted to float (e.g. "N/A")
                if nan_flag[i][v] == False:
                    data_arr[i][v] = int(last_row[v])
                    nan_flag[i][v] = True
                else:
                    try:
                        data_arr[i][v] = int(round(value_from_mean_week(mean_weak_df, "H"+str(v+1), timeStamp)))
                    except:
                        data_arr[i][v] = int(0)

        dataset[i].loc[len(dataset[i])] = data_arr[i]
    return data_arr, nan_flag, dataset

# get predictions
def predictions(df, tcID, modelsPATH, queue_var):
    # Load all models and scalers first
    models = {}
    scalers = {}

    for col in df.columns:
        predict = queue_var[str(tcID)][col]["predict"]

        if predict == "True":
            scaler_path = os.path.join(modelsPATH, f"tcID{tcID}{col}_scaler.pkl")
            model_path = os.path.join(modelsPATH, f"tcID{tcID}{col}_model.h5")

            # Load the saved scaler and model if not loaded yet
            if col not in scalers:
                with open(scaler_path, "rb") as file:
                    scalers[col] = pickle.load(file)

            if col not in models:
                models[col] = load_model(model_path)

    # Prepare input data for all columns at once using a vectorized approach
    input_data = {}
    for col in df.columns:
        predict = queue_var[str(tcID)][col]["predict"]

        if predict == "True":
            my_array = df[col].values.reshape((-1, 1))  # Reshape to (-1, 1)
            my_array = scalers[col].transform(my_array)

            # Reshape the input array to shape (len(df)-288+1, 288, 1)
            input_data_col = []
            for i in range(len(df) - 288 + 1):
                input_data_col.append(my_array[i:i + 288, :])

            input_data[col] = np.array(input_data_col)

    # Make predictions in parallel for each column
    predictions = []
    for col in df.columns:
        predict = queue_var[str(tcID)][col]["predict"]

        if predict == "True":
            model = models[col]
            input_data_col = input_data[col]

            # Make predictions for each input_data_col slice
            predictions_scaled = model.predict(input_data_col)
            prediction = scalers[col].inverse_transform(predictions_scaled)
            prediction[prediction < 0] = np.nan  # Set negative predictions to NaN

            # Append the first prediction to the result list
            predictions.append(int(round(prediction[0][0])))
        else:
            predictions.append(np.nan)

    del model, input_data

    return predictions

def perform_predictions(data_arr, tcID, dataset, config, modelsPATH):
    predData = []
    pool = multiprocessing.Pool(processes=4)

    results = [pool.apply_async(predictions, (dataset[i], tcID[i], modelsPATH, config)) for i in range(len(tcID))]
    predData = [result.get() for result in results]

    pool.close()
    pool.join()
    return predData

# Upload to database
def firebase_log(timestamp, log_array, tittle_array, project, tcID, log_type):
    # Reference to the Firebase node where you want to store the data
    ref = db.reference("logs")  # Reference to the Firebase node where you want to store the data
    year = timestamp.strftime("%Y")
    month = timestamp.strftime("%m")
    day = timestamp.strftime("%d")
    time = timestamp.strftime("%H-%M")
    log_data = dict(zip(tittle_array, log_array))  # Create a dictionary from titleArr and logArr

    # Push data to Firebase with the timestamp as the key
    ref.child(project).child(log_type).child(year).child(month).child(day).child(time).child("tcID"+str(tcID)).set(log_data)

    return

def firebase_startup_c(cycle, project):
    # Reference to the Firebase node where you want to store the data
    ref = db.reference("configuration")  # Reference to the Firebase node where you want to store the data
    #startup = dict(zip(cycle))  # Create a dictionary from titleArr and logArr

    # Push data to Firebase with the timestamp as the key
    ref.child(project).child("startup_c").set(cycle)

    return


def log_data_to_firebase(current_timestamp, data_arr, predData, config, tcID, project):
    """
    Logs traffic data to Firebase for each traffic control ID (tcID).

    Parameters:
    - current_timestamp (datetime): The current timestamp for the log entry.
    - data_arr (list of lists): Processed traffic data for each tcID.
    - predData (list of lists): Prediction data corresponding to traffic data.
    - config (dict): Configuration dictionary containing logging settings.
    - tcID (list): List of traffic control IDs.
    - project (str): Project name or identifier for Firebase logging.

    Returns:
    - None
    """
    for i in range(len(tcID)):
        logArr = [current_timestamp.strftime("%Y.%m.%d %H:%M")]
        titleArr = ["timeStamp"]

        for n in range(len(data_arr[i])):
            logArr.extend([data_arr[i][n], predData[i][n]])
            titleArr.extend([f"H{n+1}", f"predH{n+1}"])

        if config["extra"]["log"]["enable"] == "True":
            try:
                firebase_log(current_timestamp, logArr, titleArr, project, tcID[i], "predictions")
            except Exception as e:
                print(f"[DEBUG] Firebase log error for tcID {tcID[i]}: {e}")
                print(f"[DEBUG] logArr = {logArr}")
                print(f"[DEBUG] predData[i] = {predData[i]}")
                breakpoint()
                raise e

def calculate_evaluation_data(data_arr, eval_data_arr, config, queued_arr):
    """
    Calculates evaluation data by combining traffic data with queued data based on the configuration.

    Parameters:
    - data_arr (list of lists): Traffic data for evaluation.
    - eval_data_arr (list of lists): Pre-allocated array to store evaluation data.
    - config (dict): Configuration dictionary containing evaluation settings.
    - queued_arr (list of lists): Data representing queued traffic.

    Returns:
    - list of lists: Updated evaluation data array.
    """
    for i in range(len(data_arr)):
        for j in range(len(data_arr[i])):
            if config["max_pressure"] == "True" and queued_arr:
                eval_data_arr[i][j] = data_arr[i][j] + queued_arr[i][j]
            else:
                eval_data_arr[i][j] = data_arr[i][j]
    return eval_data_arr

# queued cars estimation for max pressure
def queued(config, tcID, ID, data, dSteps, c_index, m):
    """
    Estimate queued cars for a specific traffic control ID.

    Parameters:
    - config (dict): Configuration dictionary containing cycles and detection settings.
    - tcID (list): List of traffic control IDs.
    - ID (str): Current traffic control ID.
    - data (list of lists): Traffic data for detectors.
    - dSteps (list of lists): Decision steps for the current cycle.
    - c_index (int): Index of the current cycle.
    - m (float): Minimum threshold.

    Returns:
    - list: List of queued cars for each detector.
    """
    try:
        # Retrieve current cycle configuration
        cycles = [config["cycles"][key] for key in config.get("cycles", {})]
        if c_index >= len(cycles):
            raise IndexError(f"Invalid cycle index: {c_index}. Available cycles: {len(cycles)}")
        
        current_cycle = cycles[c_index]

        # Map detector and decision step data
        detectors = {
            f"tcID{tcID[i]}H{j+1}": data[i][j]
            for i in range(len(data))
            for j in range(len(data[i]))
        }

        decision_steps = {
            f"tcID{tcID[i]}dStep{j+1}": dSteps[c_index][i][j]
            for i in range(len(dSteps[c_index]))
            for j in range(len(dSteps[c_index][i]))
        }

        # Evaluation context for eval
        eval_context = {
            **detectors,
            **decision_steps,
            'm': m,
            'C': current_cycle  # Pass the current cycle explicitly
        }

        # Calculate queued cars
        numH = config.get(str(ID), {}).get("numdet", 0)
        if not isinstance(numH, int):
            raise ValueError(f"'numdet' should be an integer, got {type(numH)}")

        q = []
        for i in range(numH):
            formula = config.get(str(ID), {}).get(f"H{i+1}", {}).get("queued", "0")
            try:
                x = round(eval(formula, {}, eval_context))
                q.append(max(x, 0))  # Ensure non-negative queued count
            except Exception as e:
                print(f"Error evaluating queued for H{i+1} (tcID {ID}): {e}")
                q.append(0)

        return q

    except (KeyError, IndexError, ValueError) as specific_error:
        print(f"Configuration or Index error in queued function: {specific_error}")
        return []
    except Exception as e:
        print(f"Unexpected error in queued function: {e}")
        return []



def calculate_queues(tcID, data_arr, eval_dsteps_arr, eval_c_index, sMin, config):
    """
    Calculate queued cars for each traffic control ID.

    Parameters:
    - tcID (list): List of traffic control IDs.
    - data_arr (list of lists): Traffic data for each detector.
    - eval_dsteps_arr (list of lists): Decision steps for each cycle.
    - eval_c_index (int): Index of the current cycle.
    - sMin (float): Minimum threshold.
    - config (dict): Configuration dictionary.

    Returns:
    - list: List of queued cars for each traffic control ID.
    """
    queued_arr = []
    
    if config.get("max_pressure") == "True" and eval_dsteps_arr and eval_c_index is not None:
        for i in range(len(tcID)):
            try:
                queued_result = queued(
                    config, tcID, tcID[i], data_arr, eval_dsteps_arr, eval_c_index, sMin
                )
                queued_arr.append(queued_result)
            except Exception as e:
                print(f"Error calculating queued cars for tcID {tcID[i]}: {e}")
                queued_arr.append([])  # Append an empty result on failure
    
    return queued_arr

def generate_prediction_paths(timestamp, project, tcID):
    """
    Generate file paths for predictions based on a timestamp, project, and task/component ID.

    Args:
        timestamp (datetime): The current time for generating paths.
        project (str): The project name to include in the paths.
        tcID (int): The unique identifier for the task/component.

    Returns:
        tuple: Paths for the previous 5-minute interval and the current timestamp.
    """
    def format_path(base_time):
        return (
            f"logs/{project}/predictions/"
            f"{base_time.year}/{base_time.month:02d}/{base_time.day:02d}/"
            f"{base_time.strftime('%H-%M')}/tcID{tcID}"
        )

    # Adjust to the nearest 5-minute interval
    previous_time = timestamp - timedelta(minutes=5)

    # Generate paths
    path_prev = format_path(previous_time)
    path_cur = format_path(timestamp)

    return path_prev, path_cur

# def get_paths_for_predictions(timestamp, project, tcID):
#     """
#     Generate file paths for predictions based on a timestamp, project, and task/component ID.

#     Args:
#         timestamp (datetime): The current time for generating paths.
#         project (str): The project name to include in the paths.
#         tcID (int): The unique identifier for the task/component.

#     Returns:
#         tuple: Paths for the previous 5-minute interval and the current timestamp.
#     """
#     def format_path(base_time):
#         return (
#             f"logs/{project}/predictions/"
#             f"{base_time.year}/{base_time.month:02d}/{base_time.day:02d}/"
#             f"{base_time.strftime('%H-%M')}/tcID{tcID}"
#         )

#     # Adjust to the nearest 5-minute interval
#     previous_time = timestamp - timedelta(minutes=5)

#     # Generate paths
#     path_prev = format_path(previous_time)
#     path_cur = format_path(timestamp)

#     return path_prev, path_cur

def get_paths_for_predictions(timestamp, project, tcID):
    # Adjust time to the nearest 5-minute interval
    adjusted_time = timestamp - timedelta(minutes=5)

    # Generate path based on the adjusted time
    path_prev = f"logs/{project}/predictions/{adjusted_time.year}/{adjusted_time.month:02d}/{adjusted_time.day:02d}/{adjusted_time.strftime('%H-%M')}/tcID{str(tcID)}"
    # Generate path based on the current time
    path_cur = f"logs/{project}/predictions/{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}/{timestamp.strftime('%H-%M')}/tcID{str(tcID)}"

    return path_prev, path_cur

def process_prediction_errors(firebase_handler, pred_dis, tcID, detectors, error_data, timestamp, project):
    """
    Processes prediction errors by comparing predicted and actual detector values retrieved from Firebase.

    Parameters:
    - firebase_handler (object): An instance of FirebaseHandler to handle Firebase operations.
    - pred_dis (int): Indicator of prediction status (0 if predictions are enabled, 1 otherwise).
    - tcID (list): List of traffic control IDs.
    - detectors (list of lists): Detector IDs for each traffic control ID.
    - error_data (dict): Dictionary storing FIFO queues and cumulative error metrics for previous and predicted errors.
    - timestamp (datetime): Current timestamp to determine paths for retrieving data.
    - project (str): Project name or identifier for Firebase paths.

    Returns:
    - dict: Updated `error_data` dictionary with new error values.
    """
    for i in range(len(tcID)):
        if pred_dis == 0:  # Proceed only if predictions are enabled
            # Get Firebase paths for current and previous predictions
            path_prev, path_cur = get_paths_for_predictions(timestamp, project, tcID[i])

            # Retrieve data from Firebase
            df_prev = firebase_handler.retrieve_data(path_prev)
            df_cur = firebase_handler.retrieve_data(path_cur)

            if df_prev is not None:  # Process data only if previous data exists
                for index, det in enumerate(detectors[i]):
                    # Retrieve predicted, previous, and current detector values
                    predicted = df_prev.get(f"pred{det}")
                    previous = df_prev.get(det)
                    current = df_cur.get(det)

                    # Calculate prediction and previous errors
                    pred_error = abs(predicted - current)
                    previous_error = abs(previous - current)

                    # Update FIFO queues and cumulative errors
                    error_data["fifo_prev_err"][i][index].append(previous_error)
                    error_data["fifo_pred_err"][i][index].append(pred_error)

                    error_data["err_pred"][i][index] += pred_error
                    error_data["err_prev"][i][index] += previous_error

                    # Ensure FIFO queues maintain a maximum length of 12
                    if len(error_data["fifo_prev_err"][i][index]) > 12:
                        error_data["err_prev"][i][index] -= error_data["fifo_prev_err"][i][index].pop(0)
                    if len(error_data["fifo_pred_err"][i][index]) > 12:
                        error_data["err_pred"][i][index] -= error_data["fifo_pred_err"][i][index].pop(0)

            # Clean up references to free memory
            del path_prev, path_cur, df_prev, df_cur

    return error_data


def adapt_and_update_traffic_data(tcID, data_arr, predData, queued_arr, error_data, config, eval_run):
    """
    Adapts and updates traffic data based on predictions, queued data, and error analysis.

    Parameters:
    - tcID (list): List of traffic control IDs.
    - data_arr (list of lists): Current traffic data for each detector.
    - predData (list of lists): Predicted traffic data for each detector.
    - queued_arr (list of lists): Queued traffic data for each detector.
    - error_data (dict): Dictionary containing prediction and previous error metrics.
    - config (dict): Configuration dictionary with detector and system settings.
    - eval_run (bool): Indicates whether the evaluation run is active.

    Returns:
    - list of lists: Updated traffic data array after adaptation.
    """
    for i in range(len(tcID)):
        # Log current counts and queued data if applicable
        print(f"Counts for tcID{tcID[i]}: {data_arr[i]}")
        if config.get("max_pressure") == "True" and eval_run:
            print(f"Queued for tcID{tcID[i]}: {queued_arr[i]}")

        for v in range(len(data_arr[i])):
            detector_config = config[str(tcID[i])][f"H{v + 1}"]
            if (
                detector_config.get("predict") == "True"
                and not math.isnan(predData[i][v])
                and error_data["err_pred"][i][v] < error_data["err_prev"][i][v]
            ):
                # Update with prediction if conditions are met
                data_arr[i][v] = predData[i][v]
                if config.get("max_pressure") == "True" and eval_run and queued_arr:
                    data_arr[i][v] += queued_arr[i][v]
            else:
                # Add queued data if applicable
                if config.get("max_pressure") == "True" and eval_run and queued_arr:
                    data_arr[i][v] += queued_arr[i][v]

        # Log predictions and adapted volumes
        try:
            print(f"Predictions for tcID{tcID[i]}: {predData[i]}")
        except Exception as e:
            print(f"Error printing predictions for tcID{tcID[i]}: {e}")

        print(f"Adapted volumes for tcID{tcID[i]}: {data_arr[i]}")

        # Log error differences if possible
        try:
            error_dif = list(map(lambda x, y: x - y, error_data["err_prev"][i], error_data["err_pred"][i]))
            print(f"Error differences for tcID{tcID[i]}: {error_dif}")
        except Exception as e:
            print(f"Error calculating error differences for tcID{tcID[i]}: {e}")

    return data_arr

def check_traffic_conditions(global_traffic, config):
    """
    Checks the traffic conditions for low and high traffic and updates the no_act flag accordingly.

    Args:
        global_traffic (list): A list of traffic volume values.
        config (dict): Configuration dictionary with low and high traffic thresholds.

    Returns:
        dict: A dictionary containing the traffic condition flags: lowTraffic, highTraffic, and no_act.
    """
    # Low traffic check
    lowTraffic = 0
    if not global_traffic:  # If the traffic list is empty
        lowTraffic = 1
    else:
        for volume in global_traffic:
            if volume < config["lowTraffic"]["volume"]:
                lowTraffic = 1
            else:
                lowTraffic = 0

    # Set no_act flag if low traffic is detected
    no_act = 1 if lowTraffic == 1 else 0

    # c_max threshold check
    c_max_mode = 0
    for volume in global_traffic:
        if volume > config["c_max"]["volume"]:
            c_max_mode = 1

    # High traffic check
    highTraffic = 0
    for volume in global_traffic:
        if volume > config["highTraffic"]["volume"]:
            highTraffic = 1

    return c_max_mode, highTraffic, no_act

# Set adaptive mode off
def adaptiveOff(upperLevel, override, adaptiveOff, timeStamp):
    if upperLevel == "True" and override == "True":
        return "override"
    for i in adaptiveOff:
        if datetime.strptime(adaptiveOff[i]["from"],"%H:%M").time() < timeStamp.time() < datetime.strptime(adaptiveOff[i]["to"],"%H:%M").time():
            return adaptiveOff[i]["PR"]

    return "adaptive"


def select_best_cycle(global_vars, highTraffic):
    """
    Select the best cycle after SUMO tries based on traffic conditions and time loss.
    """
    if highTraffic == 0:
        # Compare with the previous best cycle
        if min(global_vars["time_loss"]) <= global_vars["time_loss"][global_vars["tryCycles"].index(global_vars["previous_best_cycle"])] * (1 - global_vars['improvement_percentage']):
            global_vars['c_sumo'].append(global_vars["tryCycles"][global_vars["time_loss"].index(min(global_vars["time_loss"]))])
        else:
            global_vars['c_sumo'].append(global_vars["previous_best_cycle"])
        print("tryCycles: ", global_vars["tryCycles"], " timeLoss: ", global_vars["time_loss"], " Selected cycle: ", global_vars['c_sumo'])
    else:
        # High traffic: select maximum cycle
        global_vars['c_sumo'].append(max(global_vars["tryCycles"]))
        print("High traffic condition activated. Maximum cycle selected")
        print("tryCycles: ", global_vars["tryCycles"], " timeLoss: ", global_vars["time_loss"], " Selected cycle: ", global_vars['c_sumo'])

def select_optimal_cycle(global_vars, c_max):
    """
    Selects the optimal traffic light cycle based on configuration and system state.

    Args:
        cycle_counter (int): Current cycle counter.
        sumo_cycles (list): List of cycles selected from previous iterations.
        previous_best_cycle (int): The previously best-selected cycle.
        available_cycles (list): List of available cycle lengths.
        c_max: cycle selection method (mean or max)

    Returns:
        tuple: Updated cycle counter, optimal cycle, reset `sumo_cycles`, and updated `previous_best_cycle`.
    """
    global_vars['cycle_counter'] -= 1
    if global_vars['cycle_counter'] == 0:
        global_vars['cycle_counter'] = 3

        if c_max == 1:
            # Select maximum cycle
            optimal_cycle = max(global_vars['c_sumo'])
        else:
            # Select mean cycle
            optimal_cycle = 0
            dummy = 0
            for i in range(len(global_vars['c_sumo'])):
                dummy = dummy + 1
                optimal_cycle = optimal_cycle + global_vars['c_sumo'][i]
            optimal_cycle = optimal_cycle/dummy
            optimal_cycle = int(10 * round(float(optimal_cycle)/10))
        global_vars['c_sumo'] = []

        # check distance for changing cycle
        if global_vars["previous_best_cycle"] > optimal_cycle:
            # Take one step back for gradual reduce
            optimal_cycle = global_vars["tryCycles"][global_vars["tryCycles"].index(global_vars["previous_best_cycle"]) - 1]
        global_vars["previous_best_cycle"] = optimal_cycle
        print("Optimal cycle selected: ", optimal_cycle)


def process_offsets_and_decision_steps(sumo_handler, tcID, config, offsetsSumo_arr, dsteps_arr, c_index):
    """Process offsets and decision steps into XML."""
    for i in range(len(tcID)):
        # Load offsets
        sumo_handler.tlstoxml_offset(tcID[i], config[str(tcID[i])]["sumoPrIDs"], offsetsSumo_arr[c_index][i])
        # Load decision steps
        k = 0
        for j in config[str(tcID[i])]["stages"]:
            sumo_handler.tlstoxml(tcID[i], config[str(tcID[i])]["sumoPrIDs"], config[str(tcID[i])]["stages"][j]["des_step"], dsteps_arr[c_index][i][k])
            k += 1

def process_algo_command_and_logs(tcID, config, global_vars, c_index, env_config, timeStamp):
    """Send algorithm commands, log the process, and return results."""
    results = []  # Store results for each ID
    for i in range(len(tcID)):
        stages = ";".join(map(str, global_vars["dsteps_arr"][c_index][i]))
        offZone = adaptiveOff(config[str(tcID[i])]["upperLevel"]["enable"], config[str(tcID[i])]["upperLevel"]["override"], config[str(tcID[i])]["adaptiveOff"], timeStamp)
        
        if offZone == "adaptive":
            send_ID = tcID[i] if config["test_mode"] == "False" else 250
            result = plan_params(global_vars["stc_token"], send_ID, 1, global_vars["previous_best_cycle"], global_vars["offsets_arr"][c_index][i], 0, stages, env_config["stcweb_url"])
            print('STCWeb2: ', result.json())
            firebase_startup_c(global_vars["previous_best_cycle"], env_config["project"])
            log_algo_command(config, tcID[i], result, timeStamp, global_vars["dsteps_arr"][c_index][i], global_vars["offsets_arr"][c_index][i], global_vars["tr_stg"][i], global_vars["global_traffic"][i], global_vars["previous_best_cycle"], "plan params", env_config["project"])
            results.append(result)
        
        elif offZone == "override":
            print(f'tcID {tcID[i]} Upper Level override')
            firebase_startup_c(global_vars["previous_best_cycle"], env_config["project"])
            log_algo_command(config, tcID[i], None, timeStamp, [], None, None, None, global_vars["previous_best_cycle"], "Upper Level override", env_config["project"])
            results.append(None)
        
        else:
            send_ID = tcID[i] if config["test_mode"] == "False" else 250
            result = plan_selection(global_vars["stc_token"], send_ID, offZone, env_config["stcweb_url"])
            print('STCWeb2: ', result.json())
            firebase_startup_c(global_vars["previous_best_cycle"], env_config["project"])
            log_algo_command(config, tcID[i], result, timeStamp, [], None, None, None, global_vars["previous_best_cycle"], "adaptive off condition", env_config["project"])
            results.append(result)
    


def log_algo_command(config, tcID, result, timeStamp, dsteps, offset, tr_stg, global_traffic, c, log_type, project):
    """Log algorithm command results to Firebase."""
    logArr = [timeStamp.strftime("%Y.%m.%d %H:%M"), c]
    logArr.extend(dsteps or [])
    if offset is not None:
        logArr.append(offset)
    if tr_stg:
        logArr.extend(tr_stg)
    if global_traffic:
        logArr.append(global_traffic)
    if result:
        logArr.append(result.json().get("connected"))
        logArr.append(result.json().get("cmdSent"))

    titleArr = ["TimeStamp", "C"]
    for j in range(len(config[str(tcID)]["stages"])):
        titleArr.append(f"des_step{j+1}")
    titleArr.append("offset")
    for j in range(len(config[str(tcID)]["stages"])):
        titleArr.append(f"tr_stg{j+1}")
    titleArr.extend(["total traffic", "connected", "cmdSent"])

    if config["extra"]["log"]["enable"] == "True":
        firebase_log(timeStamp, logArr, titleArr, project, tcID, log_type)


def handle_low_traffic_condition(global_vars, tcID, config, env_config, timeStamp):
    """Handle low traffic condition."""
    print("low traffic condition. controllers fall back to local mode")

    # Initialize variables
    global_vars['cl'] = 3
    global_vars['c_sumo'] = []
    global_vars['startup_cycle'] = config["cycles"]["c1"]
    global_vars['eval_run'] = False
    global_vars['eval_no_act'] = 0
    global_vars['eval_c_index'] = 0

    # Send commands to controllers to fall back into local mode
    for i in range(len(tcID)):
        send_ID = tcID[i] if config["test_mode"] == "False" else 250
        result = plan_selection(global_vars["stc_token"], send_ID, 42, env_config['stcweb_url'])
        print('STCWeb2: ', result.json())
        firebase_startup_c(global_vars['startup_cycle'], env_config['project'])

        # Log
        logArr = [timeStamp.strftime("%Y.%m.%d %H:%M"), "low traffic condition"]
        titleArr = ["TimeStamp", "C"]
        for j in range(len(config[str(tcID[i])]["stages"])):
            titleArr.append("des_step" + str(j + 1))
        titleArr.append("offset")
        for j in range(len(config[str(tcID[i])]["stages"])):
            titleArr.append("tr_stg" + str(j + 1))
        titleArr.extend(["total traffic", "connected", "cmdSent"])

        if config["extra"]["log"]["enable"] == "True":
            firebase_log(timeStamp, logArr, titleArr, env_config['project'], tcID[i], "plan params")

        del logArr, titleArr

    return global_vars

def select_fixed_time_plan_junctions(config, stc_token, c, stcweb_url):
    """
    Select and execute a fixed-time plan for specified junctions.

    Parameters:
        config (dict): Configuration dictionary containing fixed-time plan data.
        stc_token (str): Authorization token for STCWeb.
        c (int): Cycle identifier.
        stcweb_url (str): URL for STCWeb.

    Returns:
        list: A list of results from the plan_params function for each junction.
    """
    results = []

    if config["fxtcID"] != "":
        pr = int(config["PR"][str(c)])
        for i in config["fxtcID"]:
            fixtoffset = round(eval(config["fxtoffsets"][i][pr]))
            fixtstages = config["fxtstages"][i][pr]
            send_ID = config["fxtcID"][i] if config["test_mode"] == "False" else 250
            result = plan_params(stc_token, send_ID, pr, c, fixtoffset, 0, fixtstages, stcweb_url)
            print('STCWeb2: ', result.json())
            results.append(result)

def save_eval_data(global_vars):
    """Save decision steps and offsets for evaluation."""
    global_vars["eval_dsteps_arr"] = [[0 for i in range(len(n))] for n in global_vars["dsteps_arr"]]
    for i in range(len(global_vars["dsteps_arr"])):
        for j in range(len(global_vars["dsteps_arr"][i])):
            global_vars["eval_dsteps_arr"][i][j] = global_vars["dsteps_arr"][i][j]

    global_vars["eval_offsetsSumo_arr"] = [[0 for i in range(len(n))] for n in global_vars["offsetsSumo_arr"]]
    for i in range(len(global_vars["offsetsSumo_arr"])):
        for j in range(len(global_vars["offsetsSumo_arr"][i])):
            global_vars["eval_offsetsSumo_arr"][i][j] = global_vars["offsetsSumo_arr"][i][j]
    

def process_sumo_data(tcID, eval_data_arr, firebase_config, sumo_handler, global_vars, sumo_data, peak_periods_dataframe, threshold_stats_dataframe):
    """Process SUMO-related data: flows, turns, parking areas, offsets, and decision steps."""
    for i in range(len(tcID)):
        # SUMO flows
        flows = sumo_handler.sumoFlows(tcID, tcID[i], eval_data_arr, peak_periods_dataframe, threshold_stats_dataframe)
        for f in flows:
            sumo_data["sumo_flows"].append(f)

        # SUMO turns
        turns = sumo_handler.sumoTurns(tcID, tcID[i], eval_data_arr, peak_periods_dataframe, threshold_stats_dataframe)
        for t in turns:
            sumo_data["sumo_turns"].append(t)

        # SUMO parking areas
        sumo_handler.closedLane(tcID, tcID[i], eval_data_arr)

        # Load offset to XML
        sumo_handler.tlstoxml_offset(
            tcID[i],
            firebase_config[str(tcID[i])]["sumoPrIDs"],
            global_vars["eval_offsetsSumo_arr"][global_vars["eval_c_index"]][i]
        )

        # Load decision steps to XML
        k = 0
        for j in firebase_config[str(tcID[i])]["stages"]:
            sumo_handler.tlstoxml(
                tcID[i],
                firebase_config[str(tcID[i])]["sumoPrIDs"],
                firebase_config[str(tcID[i])]["stages"][j]["des_step"],
                global_vars["eval_dsteps_arr"][global_vars["eval_c_index"]][i][k]
            )
            k += 1


def log_sumo_evaluation_results(sumoStat, firebase_config, firebase_handler, timeStamp, env_config):
    """Log SUMO evaluation results."""
    logArr_sumo = [timeStamp.strftime("%Y.%m.%d %H:%M"), sumoStat[0], sumoStat[1], sumoStat[2]]
    titleArr_sumo = ["timeStamp", "timeLoss", "speed", "duration"]
    if firebase_config["extra"]["log"]["enable"] == "True":
        log_type = "Evaluation"
        firebase_handler.log_data(
            timeStamp,
            logArr_sumo,
            titleArr_sumo,
            env_config["project"],
            "",
            log_type
        )


def evaluate_sumo_run(no_act, firebase_config, global_vars, tcID, eval_data_arr, sumo_handler, firebase_handler, timeStamp, env_config, sumo_data, c_index, peak_periods_dataframe, threshold_stats_dataframe):
    """Evaluate the SUMO run based on global variables and configurations."""
    if not global_vars["eval_run"]:
        global_vars["eval_c_index"] = c_index
        save_eval_data(global_vars)
        global_vars["eval_no_act"] = no_act
        global_vars["eval_run"] = True
    else:
        if global_vars["eval_no_act"] == 0 and firebase_config["extra"]["eval"] == "True":
            process_sumo_data(tcID, eval_data_arr, firebase_config, sumo_handler, global_vars, sumo_data, peak_periods_dataframe, threshold_stats_dataframe)

            # Load flows and turns to XML
            sumo_handler.flowtoxml(sumo_data["sumo_flows"])
            sumo_handler.turnstoxml(sumo_data["sumo_turns"])

            # Run SUMO
            sumoStat = runSUMO()

            # Log SUMO results
            log_sumo_evaluation_results(sumoStat, firebase_config, firebase_handler, timeStamp, env_config)

        save_eval_data(global_vars)
        global_vars["eval_no_act"] = no_act
        global_vars["eval_c_index"] = c_index


def compute_mean_week(dataframe, tc_id=None, save_path=None):
    """
    Computes the mean weekly pattern of detector values for a DataFrame.

    Parameters:
    - dataframe: pd.DataFrame with columns ['TimeStamp', 'H1', ..., 'HN']
    - tc_id: Optional, used to save the result with a file name like mean_week_tcID{tc_id}.xlsx
    - save_path: Optional, if provided saves the result as Excel file

    Returns:
    - DataFrame: Mean weekly pattern
    """
    df = dataframe.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df = df.set_index('TimeStamp')

    df_weekday = df.groupby([df.index.hour, df.index.weekday, df.index.minute]).mean()
    df_weekday["day"] = df_weekday.index.get_level_values(1)
    df_weekday["hour"] = df_weekday.index.get_level_values(0)
    df_weekday["minute"] = df_weekday.index.get_level_values(2)

    df_weekday = df_weekday.sort_values(by=["day", "hour", "minute"])
    df_weekday["day"] = df_weekday["day"].replace({
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday"
    })

    df_weekday = df_weekday.reset_index(drop=True)

    if save_path and tc_id is not None:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"mean_week_tcID{tc_id}.xlsx")
        df_weekday.to_excel(file_path, index=False)

    return df_weekday

def perform_predictions_transformers(dataset, tcID, firebase_config, handlers):
    """
    Perform predictions using trained Transformer models.

    Parameters:
    - dataset: list of DataFrames, one per tcID (each with detector columns only, no timestamp)
    - tcID: list of traffic control IDs
    - firebase_config: configuration dictionary with 'predict' flags
    - handlers: dict of TransformerTimeSeriesHandler instances keyed by tcID (e.g., 'tcID6')

    Returns:
    - predData: list of lists containing predictions per tcID and detector
    """
    predData = []

    for i in range(len(tcID)):
        df = dataset[i].copy()
        preds = []

        # 🔧 Inject synthetic TimeStamp column assuming 5-minute intervals
        df_length = len(df)
        end_time = datetime.now()
        df["TimeStamp"] = [end_time - timedelta(minutes=5 * (df_length - j - 1)) for j in range(df_length)]

        for col in df.columns:
            if col == "TimeStamp":
                continue

            if firebase_config[str(tcID[i])][col]["predict"] == "True":
                try:
                    # Use df with synthetic TimeStamp
                    pred = handlers[f"tcID{tcID[i]}{col}"].predict(df=df[["TimeStamp", col]])
                    if pred < 0:
                        pred = 0
                except Exception as e:
                    print(f"[ERROR] Prediction failed for tcID {tcID[i]}, detector {col}: {e}")
            else:
                pred = df[[col]].iloc[-1][0]

            preds.append(pred)

        predData.append(preds)

    return np.array(predData)

def should_retrain_now(retrain_time):
    now = datetime.now()
    return (
        now.weekday() == retrain_time["weekday"] and
        now.hour == retrain_time["hour"] and
        00 <= now.minute <= 5  # between :00 and :05
    )