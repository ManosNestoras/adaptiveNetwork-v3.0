"""
Traffic Technique - Adaptive Control Module

Authors: Georgios Terzoglou, Manos Nestwras  
Organization: Traffic Technique  
Date: 2024-12-29  

Description:  
This module handles interactions with Firebase, including data retrieval, logging, and configuration management.  
It serves as the interface between the Adaptive Network Control system and Firebase Realtime Database.  
"""



from firebase_admin import credentials, db
import firebase_admin
from datetime import timedelta

class FirebaseHandler:
    """Handles Firebase interactions for logging, data retrieval, and configuration management."""
    def __init__(self, paths):
        """Initializes the Firebase connection."""
        cred = credentials.Certificate(paths['filePATH'] + paths['Firebase_key'])
        firebase_admin.initialize_app(cred, {"databaseURL": paths['Firebase_url']})

    def log_data(self, timestamp, log_array, title_array, project, tcID, log_type):
        """Logs data to Firebase under a specific node."""
        try:
            ref = db.reference("logs")
            log_data = dict(zip(title_array, log_array))
            year, month, day, time = timestamp.strftime("%Y %m %d %H-%M").split()
            ref.child(project).child(log_type).child(year).child(month).child(day).child(time).child(f"tcID{tcID}").set(log_data)
        except Exception as e:
            print(f"Failed to log data to Firebase: {e}")

    def startup_config(self, cycle, project):
        """Sets the startup configuration in Firebase."""
        try:
            ref = db.reference("configuration")
            ref.child(project).child("startup_c").set(cycle)
        except Exception as e:
            print(f"Failed to set startup configuration: {e}")

    def restart_config(self, project):
        """Resets the restart configuration in Firebase."""
        try:
            ref = db.reference("configuration")
            ref.child(project).child("restart").set(0)
        except Exception as e:
            print(f"Failed to reset configuration: {e}")
    
    def update_config(self, project):
        """Resets the update configuration in Firebase."""
        try:
            ref = db.reference("configuration")
            ref.child(project).child("update").set(0)
        except Exception as e:
            print(f"Failed to reset update configuration: {e}")

    def get_paths_for_predictions(self, timestamp, project, tcID):
        """Generates paths for retrieving prediction data based on the timestamp."""
        # Adjust time to the nearest 5-minute interval
        adjusted_time = timestamp - timedelta(minutes=5)

        # Generate paths based on adjusted time and current time
        path_prev = f"logs/{project}/predictions/{adjusted_time.year}/{adjusted_time.month:02d}/{adjusted_time.day:02d}/{adjusted_time.strftime('%H-%M')}/tcID{str(tcID)}"
        path_cur = f"logs/{project}/predictions/{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}/{timestamp.strftime('%H-%M')}/tcID{str(tcID)}"

        return path_prev, path_cur

    def retrieve_data(self, path):
        """Retrieves data from Firebase at the specified path."""
        try:
            ref = db.reference(path)
            return ref.get()
        except Exception as e:
            print(f"Failed to retrieve data from Firebase: {e}")
            return None