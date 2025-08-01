
# Adaptive Network Control Module for SCAE STCWeb2 UTC

## Setting Up the Environment

### 1. Create the Conda Environment
```bash
conda create --name adaptiveNetwork pip
```

### 2. Activate the Environment
```bash
conda activate adaptiveNetwork
```

### 3. Install Python Version 3.10.8
```bash
conda install python==3.10.8
```

### 4. Reinstall Pip
```bash
python -m pip install --upgrade --force-reinstall pip
```

### 5. Install Required Python Libraries
```bash
pip install pymongo
pip install firebase-admin
pip install tensorflow==2.12.0
pip install scikit-learn
pip install pandas
pip install python-dotenv==0.19.2
pip install flask_pymongo
pip install openpyxl
```

---

## Configuring the Environment

### Add Environmental Variables
- Insert project name, paths, credentials, and SUMO files into a `.env` file.

---

## System Configuration

### Firebase Configuration
- Add Firebase credentials and configuration variables in `upload_firebase.py`:
```python
data = {
    "adtcID": "IDs of adaptive controllers",
    "cycles": "Pool of cycles",
    "fxtcID": "IDs of fixed-time controllers",
    "PR": "Program IDs to cycle matching",
    "fxtoffsets": "Offsets of fixed-time controllers",
    "fxtstages": "Stages of fixed-time controllers",
    "startup_c": "Startup cycle (saved at the end of every run)",
    "lowTraffic": "Low traffic volume to switch to local mode",
    "highTraffic": "High traffic volume to force maximum cycle",
    "loopOffset": "Start delay",
    "c_max_method": "Implement max or median cycle method",
    "max_pressure": "Implement not-served-traffic method",
    "restart": "Restart module",
    "test_mode": "Force test method (does not send commands to controllers)",
    
    "traffic controller number": {
        "numdet": 9,  # Number of detectors
        "H1": {
            "detectorSetback": 120,
            "multilane": "True",
            "mainRoad": "True",
            "predict": "True",
            "queued": "tcID4H1-(tcID4dStep1+12-3)*(m*60/C)/2"
        },
        "stages": {
            "stage1": {
                "des_step": 1,
                "fixed_green": 6,
                "minTime": 3,
                "lost_green": 0,
                "trs_sec": 2
            }
        },
        "dead_time": 26,  # Yellow and red time
        "110": {"cycle": 110, "offset": "4", "sumoOffset": "0"},
        "sumoPrIDs": {"Pr1": 1},  # Adaptive program ID
        "sumoFlows": {
            "ID4_H1_H2": "tcID4H1+tcID4H2"
        },
        "sumoTurns": {
            "turn1": {"from": "TURNS_TO_H8_OR_TO_H4", "to": "E64", "count": "0.25*(tcID4H1+tcID4H2)"},
            "turn10": {"from": "RIGHT_TURN_TO_MIAOULI", "to": "E55", "count": "0.03*(tcID4H5+tcID4H6-tcID4H7)"}
        },
        "sumoPark": {
            "pa_0": {
                "activate": "1 if (float(tcID4H9)<0.30*tcID4H3 and tcID4H3>12) else 0",
                "edge_stop": "E41",
                "edge_go": "ID4_H3_H9"
            }
        },
        "tr_stages": {
            "tr_stg1": "max(0.60*(tcID4H1+tcID4H2)-50, 0.50*(tcID4H5+tcID4H6)-300)"
        },
        "adaptiveOff": {
            "TZone1": {"from": "02:00", "to": "02:00", "PR": 2}
        },
        "upperLevel": {"enable": "False", "override": "False"}
    },

    "extra": {
        "log": {"enable": "True"},  # Enable logs
        "eval": "True"  # Enable evaluation run
    }
}
```

---

## Upload Configuration to Firebase
Run the `upload_firebase.py` script to upload the configuration.

---

## Starting the Module

### Use a Batch File to Start
Create a `start.bat` file with the following content:
```bash
call conda activate adaptiveNetwork
python adaptiveNetwork.py
call conda deactivate
```
