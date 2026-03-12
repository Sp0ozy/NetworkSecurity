import numpy as np
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import yaml
import pickle

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """
    try:
        logging.info(f"Reading YAML file: {file_path}")
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a dictionary to a YAML file.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                logging.info(f"Replacing existing YAML file: {file_path}")
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
        logging.info(f"YAML file written: {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Saves a numpy array to a file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)
        logging.info(f"Numpy array saved to {file_path}  (shape: {array.shape})")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a numpy array from a file.
    """
    try:
        logging.info(f"Loading numpy array from {file_path}")
        with open(file_path, 'rb') as file:
            arr = np.load(file)
        logging.info(f"Numpy array loaded with shape {arr.shape} from {file_path}")
        return arr
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object of type {type(obj).__name__} saved to {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using pickle.
    """
    try:
        logging.info(f"Loading object from {file_path}")
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logging.info(f"Object of type {type(obj).__name__} loaded from {file_path}")
        return obj
    except Exception as e:
        raise NetworkSecurityException(e, sys)