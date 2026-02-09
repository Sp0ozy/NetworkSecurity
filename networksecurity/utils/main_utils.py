from networksecurity.exceptions.exception import NetworkSecurityError
from networksecurity.logging.logger import logging
import os, sys
import yaml

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityError(e, sys)