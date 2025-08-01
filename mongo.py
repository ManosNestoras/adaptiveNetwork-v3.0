"""
Traffic Technique - Adaptive Control Module

Authors: Georgios Terzoglou, Manos Nestwras  
Organization: Traffic Technique  
Date: 2024-12-29 

Description:  
This module manages MongoDB connections and handles data retrieval and aggregation.  
It processes traffic data and performs database queries efficiently for adaptive traffic control.  
"""


from flask_pymongo import pymongo
from datetime import timedelta
import pandas as pd
from utils import queue_calc

class MongoDBHandler:
    """Handles interactions with MongoDB for traffic data processing."""
    def __init__(self, mongo_url, db_name="his_1234"):
        """Initializes the MongoDB handler.

        Args:
            mongo_url (str): MongoDB connection URL.
            db_name (str): Name of the database to connect to.
        """
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        """Establishes a connection to the MongoDB database."""
        if not self.client:
            self.client = pymongo.MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]

    def disconnect(self):
        """Closes the connection to the MongoDB database."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    def get_traffic_data(self, tcID, interval, timestamp, collection_name="traffic_data"):
        # Should work like proccess volumes
        return 

    def get_5min_traffic_data(self, tcID, timestamp, collection_name="traffic_data", num_values=None):
        """
        Returns traffic data for a given tcID over a flexible time window.

        You can provide:
        - `minutes`: retrieves data going back by that many minutes.
        - `months`: retrieves data from that many months ago.
        - `num_values`: retrieves exactly that many 5-minute intervals (most recent).

        Priority: num_values > minutes > months
        """
        self.connect()
        col = self.db[collection_name]

        # Calculate time_fetch based on input
        if num_values is not None:
            minutes = num_values * 5
            time_fetch = timestamp - timedelta(minutes=minutes)
        elif minutes is not None:
            time_fetch = timestamp - timedelta(minutes=minutes)
        else:
            raise ValueError("Provide at least one of: num_values, minutes, or months.")

        query = [
            {"$match": {
                "tcID": tcID,
                "Data.Year": {"$gte": time_fetch.year % 100},
                "Data.Month": {"$gte": time_fetch.month},
                "Data.Day": {"$gte": time_fetch.day}
            }},
            {"$unwind": "$Data"},
            {"$project": {
                "data": {
                    "$map": {
                        "input": {"$range": [0, "$Data.N_CountOccu"]},
                        "in": {
                            "i": {"$add": ["$$this", 0]},
                            "counters": {"$arrayElemAt": ["$Data.CountOccu", "$$this"]},
                            "year": {"$add": [0, "$Data.Year"]},
                            "month": {"$add": [0, "$Data.Month"]},
                            "days": {"$add": [0, "$Data.Day"]},
                            "hour": {"$add": [0, "$Data.Hour"]},
                            "minutes": {"$add": [0, "$Data.Minutes"]}
                        }
                    }
                }
            }},
            {"$unwind": "$data"},
            {"$sort": {"_id": 1}},
            {"$unset": ["_id", "data.counters.speed"]}
        ]

        result = list(col.aggregate(query, allowDiskUse=True))
        self.disconnect()

        # Normalize and format
        df = pd.json_normalize(result)
        df['TimeStamp'] = pd.to_datetime(
            df[['data.days', 'data.month', 'data.year', 'data.hour', 'data.minutes']]
            .astype(str).agg(' '.join, axis=1),
            format='%d %m %y %H %M'
        )

        df.drop(columns=["data.year", "data.month", "data.days", "data.hour", "data.minutes"], inplace=True)
        df["Detectors"] = "H" + (df["data.i"] + 1).astype(str)
        df.drop(columns=["data.i"], inplace=True)

        grouped = df.groupby([pd.Grouper(key="TimeStamp", freq="5Min"), "Detectors"]).agg({
            "data.counters.count": "sum",
            "data.counters.occupancy": "mean"
        })

        grouped["Volume"] = grouped.apply(
            lambda x: queue_calc(x["data.counters.count"], x["data.counters.occupancy"]), axis=1
        )

        result_df = grouped["Volume"].unstack().rename_axis(columns=None).reset_index()
        result_df.sort_values(by="TimeStamp", inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        if num_values is not None:
            result_df = result_df[-num_values:].reset_index(drop=True)

        return result_df