import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from openpyxl import Workbook


cred = credentials.Certificate("C:/Users/Manos Nestoras/OneDrive/Aerodromio_algo/service_account_key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://adaptivenetwork-e3a22-default-rtdb.europe-west1.firebasedatabase.app/"
})


# Create excel from firebase
def create_excel_from_firebase(file_name, output_file):
    ref = db.reference("logs")

    # Retrieve data from Firebase for the given "file" key
    data = ref.child(file_name).get()

    if data is not None:
        # Create a new Excel workbook
        wb = Workbook()
        ws = wb.active

        # Add the header row
        title_array = list(data[list(data.keys())[0]].keys())
        ws.append(title_array)

        # Add data rows
        for timestamp, log_data in data.items():
            ws.append(list(log_data.values()))

        # Save the Excel file
        wb.save(f"{file}.xlsx")
        print(f"Successfully saved to {file}.xlsx")

    else:
        print(f"Data for the file '{file}.xlsx' not found in Firebase.")

def firebase_upload(data, project, tcID,):
    ref = db.reference("configuration/" + project)  # Reference to the Firebase node where you want to store the data
    ref.update(data)

def firebase_retrieve_data(path):
    ref = db.reference(path)  # Reference to the Firebase node where you want to retrieve data
    data = ref.get()
    return data

# Define the data you want to upload
data = {
    "adtcID": {"AERODROMIO": 78},
    "fxtcID": {},  
    "cycles": {"c1": 80, "c2": 90, "c3": 100, "c4": 110, "c5": 120},
    "pr": {"80": 5, "90": 4, "100": 3, "110": 2, "120": 1},
    "startup_c": 90,
    "lowTraffic": {"volume": 250},
    "highTraffic": {"volume": 2000},
    "loopOffset": 0,
    "c_max_method": "True",
    "max_pressure": "True",

    "78": {
        "numdet": 11,
        "H1": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "0"},
        "H2": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H2-(tcID78dStep1+1-3)*(m*60/C)/2"},
        "H3": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H3-(tcID78dStep1+1-3)*(m*60/C)/2"},
        "H4": {"detectorSetback": 100, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H4-(tcID78dStep4+13-3)*(m*60/C)/2"},
        "H5": {"detectorSetback": 100, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H5-(tcID78dStep4+13-3)*(m*60/C)/2.1"},
        "H6": {"detectorSetback": 100, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H6-(tcID78dStep4+13-3)*(m*60/C)/2.1"},
        "H7": {"detectorSetback": 40, "multinane": "False", "mainRoad": "False", "predict": "False", "queued": "tcID78H7-(tcID78dStep2+5-3)*(m*60/C)/2.2"},
        "H8": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H8-(tcID78dStep1+tcID78dStep2+tcID78dStep4+29-3)*(m*60/C)/2.2"},
        "H9": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H9-(tcID78dStep1+tcID78dStep2+11-3)*(m*60/C)/2"},
        "H10": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H10-(tcID78dStep3-3)*(m*60/C)/2"},
        "H11": {"detectorSetback": 120, "multinane": "True", "mainRoad": "True", "predict": "False", "queued": "tcID78H11-(tcID78dStep3-3)*(m*60/C)/2"},
        "stages":{
            "stage1": {"des_step": 0, "fixed_green": 1, "minTime": 24, "lost_green": 3, "trs_sec": 2},
            "stage2": {"des_step": 5, "fixed_green": 5, "minTime": 3, "lost_green": 3, "trs_sec": 2.2},
            "stage3": {"des_step": 11, "fixed_green": 0, "minTime": 8, "lost_green": 3, "trs_sec": 2},
            "stage4": {"des_step": 16, "fixed_green": 13, "minTime": 1, "lost_green": 3, "trs_sec": 2.1}
        },
        "dead_time": 21,
        "120": {"cycle": 120, "offset": "0", "sumoOffset": "0"},
        "110": {"cycle": 110, "offset": "0", "sumoOffset": "0"},
        "100": {"cycle": 100, "offset": "0", "sumoOffset": "0"},
        "90": {"cycle": 90, "offset": "0", "sumoOffset": "0"},
        "80": {"cycle": 80, "offset": "0", "sumoOffset": "0"},
        "sumoPrIDs": {"Pr1": 1},
        "sumoFlows": {
            "ID78_H2_H3": "tcID78H2+tcID78H3",
            "ID78_H4_H5": "tcID78H4+tcID78H5+tcID78H6",
            "ID78_H8_H9": "tcID78H8+tcID78H9",
            "ID78_H10_H11": "tcID78H10+tcID78H11"
            },
        "sumoTurns": {
            "turn1": {"from": "ID78_RIGHT_TURN_H4", "to": "ID78_RIGHT_TURN_H4.17", "count": "tcID78H4"},
            "turn2": {"from": "ID78_LEFT_TURN_H6", "to": "533831198", "count": "tcID78H5+tcID78H6"},
            "turn3": {"from": "ID78_LEFT_TURN_H7", "to": "59243444", "count": "0.75*tcID78H8"},
            "turn4": {"from": "ID78_LEFT_TURN_H7", "to": "176962956", "count": "tcID78H7"},
            "turn5": {"from": "ID78_RIGHT_TURN_H10", "to": "ID78_RIGHT_TURN_H10.29", "count": "0.05*tcID78H10"},
            "turn6": {"from": "ID78_LEFT_TURN_H11", "to": "533831210", "count": "0.5*(tcID78H10+tcID78H11)"}
            },
        "sumoPark": {
            },
        "tr_stages": {
            "tr_stg1": "max(0.5*(tcID78H2+tcID78H3), tcID78H9)",
            "tr_stg2": "tcID78H7",
            "tr_stg3": "0.5*(tcID78H10+tcID78H11)",
            "tr_stg4": "max(0.5*(tcID78H5+tcID78H6), (tcID78H8-tcID78H9-tcID78H7-150), tcID78H4)"
            },
        "adaptiveOff": {
            }
        },

    "extra": {
        "log": {"project": "PKM-Aerodromio", "enable": "True"},
        "eval": "True"
            }
}

# Specify excel file
file =""

project = "PKM-Aerodromio"
tcID = 78

# Call the firebase_upload function to upload the data
firebase_upload(data, project, tcID)

# Call the firebase_retrieve_data function to retrieve data
retrieved_data = firebase_retrieve_data("configuration/" + project)

# Now, 'retrieved_data' contains the data you retrieved from Firebase
print(retrieved_data["78"]["80"])
