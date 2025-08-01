from utils import * 
from firebase import FirebaseHandler  
from mongo import MongoDBHandler  
from sumo import SUMOHandler 
from transformersHandler import TransformerTimeSeriesHandler

import torch
import psutil

def get_safe_device(min_free_gpu_gb=1.0):
    """
    Returns 'cuda' if GPU available and has enough free memory.
    Falls back to 'cpu' otherwise.
    """
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = total - allocated
        free_gb = free / 1024**3

        if free_gb >= min_free_gpu_gb:
            print(f"✅ Using GPU | Free: {free_gb:.2f} GB")
            return torch.device("cuda")
        else:
            print(f"⚠️ GPU memory low ({free_gb:.2f} GB). Using CPU.")
            return torch.device("cpu")
    else:
        print("⚠️ No GPU found. Using CPU.")
        return torch.device("cpu")



env_config = load_config()
firebase_handler = FirebaseHandler(env_config['paths'])
mongodb_handler = MongoDBHandler(env_config['paths']['mongo_url'])
firebase_config = firebase_handler.retrieve_data("configuration/" + env_config["project"])
sumo_handler = SUMOHandler(firebase_config, env_config)
tcID, detlist = load_intersection_ids(firebase_config)

timestamp = datetime.now()
model_dir = env_config['paths']['trans_modelsPATH']

device = get_safe_device(min_free_gpu_gb=1.0)
transformer_handler = TransformerTimeSeriesHandler(model_dir=model_dir, epochs=50)
transformer_handler.device = device

for i in range(len(tcID)):
    temp = mongodb_handler.get_5min_traffic_data(tcID[i], timestamp, num_values=34560)  # 4 months

    for col in temp.columns:
        if col == "TimeStamp":
            continue
        df = temp[["TimeStamp", col]].copy()
        name = f"tcID{tcID[i]}{col}"
        transformer_handler.train_on_dataframe(df, col, name, test_days=7)
