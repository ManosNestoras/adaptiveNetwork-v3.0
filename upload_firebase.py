import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import os
import dotenv
from dotenv import load_dotenv


cred = credentials.Certificate("service_account_key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://adaptivenetwork-e3a22-default-rtdb.europe-west1.firebasedatabase.app/"
})

# load enviroment parameters
dotenv_file = dotenv.find_dotenv()
load_dotenv(dotenv_file)
project = os.getenv('project')


def firebase_upload(data, project):
    ref = db.reference("configuration/" + project)  # Reference to the Firebase node where you want to store the data
    ref.update(data)

def firebase_retrieve_data(path):
    ref = db.reference(path)  # Reference to the Firebase node where you want to retrieve data
    data = ref.get()
    return data

# Define the data you want to upload
data = {
    "adtcID": {"M_ANTYPA K1": 6, "M_ANTYPA K2": 7},  
    "cycles": {"c1": 70, "c2": 80, "c3": 90, "c4": 100},
    "fxtcID": "",
    "pr": {"70": "2", "80": "1", "90": "3", "100": "3"},
    "fxtoffsets": "",
    "fxtstages": "",
    "startup_c": 100,
    "lowTraffic": {"volume": 400},
    "c_max": {"volume": 1400},
    "highTraffic": {"volume": 2000},
    "loopOffset": 5,
    "max_pressure": "True",
    "restart": 0,
    "test_mode": "True",

    "6": {
        "numdet": 8,
        "H1": {"detectorSetback": 90, "multinane": "True", "mainRoad": "True", "predict": "True", "queued": "tcID6H1-tcID6H8-(tcID6dStep1+tcID6dStep2+6-3)*(m*60/C)/2"},
        "H2": {"detectorSetback": 90, "multinane": "True", "mainRoad": "True", "predict": "True", "queued": "tcID6H2-(tcID6dStep1+tcID6dStep2+6-3)*(m*60/C)/2"},
        "H3": {"detectorSetback": 85, "multinane": "True", "mainRoad": "True", "predict": "True", "queued": "tcID6H3-(tcID6dStep2+tcID6dStep3+8-3)*(m*60/C)/2"},
        "H4": {"detectorSetback": 85, "multinane": "True", "mainRoad": "True", "predict": "True", "queued": "tcID6H4-tcID6H5-(tcID6dStep2+tcID6dStep3+8-3)*(m*60/C)/2"},
        "H5": {"detectorSetback": 30, "multinane": "False", "mainRoad": "False", "predict": "True", "queued": "tcID6H5-(tcID6dStep3+2-3)*(m*60/C)/2.2"},
        "H6": {"detectorSetback": 70, "multinane": "True", "mainRoad": "False", "predict": "True", "queued": "tcID6H6-(tcID6dStep3+tcID6dStep4+10-3)*(m*60/C)/2.3"},
        "H7": {"detectorSetback": 70, "multinane": "True", "mainRoad": "False", "predict": "True", "queued": "tcID6H7-(tcID6dStep4+2-3)*(m*60/C)/2.2"},
        "H8": {"detectorSetback": 5, "multinane": "False", "mainRoad": "False", "predict": "True", "queued": "tcID6H8-(tcID6dStep1+tcID6dStep2+tcID6dStep4+10-6)*(m*60/C)/2"},
        "stages":{
            "stage1": {"des_step": 1, "fixed_green": 1, "minTime": 7, "lost_green": 3, "trs_sec": 2.2},
            "stage2": {"des_step": 4, "fixed_green": 0, "minTime": 1, "lost_green": 0, "trs_sec": 2},
            "stage3": {"des_step": 9, "fixed_green": 2, "minTime": 6, "lost_green": 3, "trs_sec": 2.3},
            "stage4": {"des_step": 15, "fixed_green": 2, "minTime": 6, "lost_green": 3, "trs_sec": 2.3},
            "stage5": {"des_step": 18, "fixed_green": 0, "minTime": 8, "lost_green": 3, "trs_sec": 2.2}
        },
        "dead_time": 26,
        "100": {"cycle": 100, "offset": "13", "sumoOffset": "0"},
        "90": {"cycle": 90, "offset": "13", "sumoOffset": "0"},
        "80": {"cycle": 80, "offset": "13", "sumoOffset": "0"},
        "70": {"cycle": 70, "offset": "13", "sumoOffset": "0"},
        "sumoPrIDs": {"Pr1": 1},
        "sumoFlows": {
            "ID6_H1+H2": "tcID6H1+tcID6H2",
            "ID6_H3+H4": "tcID6H3+tcID6H4",
            "ID6_H6+H7": "tcID6H6+tcID6H7",
            "ENTRY_FROM_APOLLONOS": "10"
            },
        "sumoTurns": {
            "turn1": {"from": "LEFT_TURN_ID6_H5", "to": "455285650", "count": "tcID6H5"},
            "turn2": {"from": "LEFT_TURN_ID6_D1", "to": "388917297", "count": "2"},
            "turn3": {"from": "455285643#1", "to": "ID6_H8", "count": "tcID6H8"}
            },
        "sumoPark": "",
        "tr_stages": {
            "tr_stg1": "0",
            "tr_stg2": "max((0.60*(tcID6H1+tcID6H2-tcID6H8)-140), (0.50*(tcID6H3+tcID6H4-tcID6H5)-80-tcID6H5))",
            "tr_stg3": "tcID6H5",
            "tr_stg4": "max(tcID6H7, (peak('tcID6H6')-tcID6H5-80))",
            "tr_stg5": "0"
            },
        "adaptiveOff": "",
        "upperLevel": {"enable": "False", "override": "False"}
        },

    "7": {
        "numdet": 7,
        "H1": {"detectorSetback": 110, "multinane": "True", "mainRoad": "True", "predict": "True", "queued": "tcID7H1-tcID7H7-(tcID7dStep1+1-3)*(m*60/C)/2"},
        "H2": {"detectorSetback": 110, "multinane": "True", "mainRoad": "True", "predict": "True", "queued": "tcID7H2-(tcID7dStep1+1-3)*(m*60/C)/2"},
        "H3": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "True", "queued": "tcID7H3-(tcID7dStep1+tcID7dStep2+7-3)*(m*60/C)/2"},
        "H4": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "True", "queued": "tcID7H4-tcID7H5-(tcID7dStep1+tcID7dStep2+7-3)*(m*60/C)/2"},
        "H5": {"detectorSetback": 55, "multinane": "False", "mainRoad": "False", "predict": "True", "queued": "tcID7H5-(tcID7dStep2-3)*(m*60/C)/2.2"},
        "H6": {"detectorSetback": 58, "multinane": "False", "mainRoad": "False", "predict": "True", "queued": "tcID7H6-(tcID7dStep3+2-3)*(m*60/C)/2.1"},
        "H7": {"detectorSetback": 5, "multinane": "False", "mainRoad": "False", "predict": "True", "queued": "tcID7H7-(tcID7dStep1+tcID7dStep3+4-6)*(m*60/C)/2.1"},
        "stages":{
            "stage1": {"des_step": 1, "fixed_green": 1, "minTime": 15, "lost_green": 3, "trs_sec": 2},
            "stage2": {"des_step": 5, "fixed_green": 0, "minTime": 7, "lost_green": 3, "trs_sec": 2.2},
            "stage3": {"des_step": 11, "fixed_green": 2, "minTime": 6, "lost_green": 3, "trs_sec": 2.1},
            "stage4": {"des_step": 15, "fixed_green": 0, "minTime": 7, "lost_green": 3, "trs_sec": 2.1}
        },
        "dead_time": 20,
        "100": {"cycle": 100, "offset": "98", "sumoOffset": "97"},
        "90": {"cycle": 90, "offset": "88", "sumoOffset": "87"},
        "80": {"cycle": 80, "offset": "78", "sumoOffset": "77"},
        "70": {"cycle": 70, "offset": "68", "sumoOffset": "67"},
        "sumoPrIDs": {"Pr1": 1},
        "sumoFlows": {
            "ID7_H1+H2": "tcID7H1+tcID7H2",
            "ID7_H3+H4": "tcID7H3+tcID7H4",
            "ID7_H6": "tcID7H6",
            "ENTRY_FROM_G_SXOLIS": "0.03*(tcID7H3+tcID7H4-tcID7H5)+0.02*(tcID7H1+tcID7H2-tcID7H7)"
            },
        "sumoTurns": {
            "turn1": {"from": "TURNS_ID7_H5", "to": "455285621", "count": "tcID7H5"},
            "turn2": {"from": "455285603#2.37.76", "to": "RIGHT_TURN_ID7_H7", "count": "tcID7H7"},
            "turn3": {"from": "TURNS_ID7_H5", "to": "322436674", "count": "0.03*(tcID7H3+tcID7H4-tcID7H5)"},
            "turn4": {"from": "LEFT_TURN_ID7_H2", "to": "322436674", "count": "0.02*(tcID7H1+tcID7H2-tcID7H7)"}
            },
        "sumoPark": "",
        "tr_stages": {
            "tr_stg1": "max((0.50*(tcID7H3+tcID7H4-tcID7H5)-100-tcID7H5), 0.50*(tcID7H1+tcID7H2-tcID6H7), (tcID6H7-tcID6H6))",
            "tr_stg2": "tcID7H5",
            "tr_stg3": "tcID7H6",
            "tr_stg4": "max(220 if timestamp.strftime('%A') in ['Monday','Tuesday','Wednesday','Thursday','Friday'] and timestamp.hour==8 and timestamp.minute<=45 else 0, 160 if timestamp.strftime('%A') in ['Monday','Tuesday','Wednesday','Thursday','Friday'] and 14<=timestamp.hour<16 else 0)"
            },
        "adaptiveOff": "",
        "upperLevel": {"enable": "False", "override": "False"}
        },

    "extra": {
        "log": {"enable": "True"},
        "eval": "True"
            }
}



# Call the firebase_upload function to upload the data
firebase_upload(data, project)

# Call the firebase_retrieve_data function to retrieve data
retrieved_data = firebase_retrieve_data("configuration/" + project)

# Now, 'retrieved_data' contains the data you retrieved from Firebase
print(retrieved_data)
