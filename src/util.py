import yaml
import joblib
from datetime import datetime

config_dir = "config/config.yaml"

def time_stamp() -> datetime:
    # Return current date and time
    return datetime.now()

def load_config() -> dict: 
    # Try to load yaml file
    try:
        with open(config_dir, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fe:
        raise RuntimeError("Parameters file not found in path.")

    # Return params in dict format
    return config

def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    # Dump data into file
    joblib.dump(data, file_path)

params = load_config()
PRINT_DEBUG = params["print_debug"]

def print_debug(messages: str) -> None:
    # Check wheter user wants to use print or not
    if PRINT_DEBUG == True:
        print(time_stamp(), messages)

