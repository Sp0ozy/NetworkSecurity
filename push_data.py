import os
import sys
import json
import certifi
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import pandas as pd
import numpy as np
from networksecurity.exceptions.exception import NetworkSecurityError
from networksecurity.logging.logger import logging


load_dotenv()
ca=certifi.where()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
FILE_PATH = "network_data/phisingData.csv"


class NetworkDataExtract():
    def __init__(self, database, collection):
        try:
            self.client = MongoClient(MONGO_DB_URL, server_api=ServerApi('1'), tlsCAFile=ca)
            self.database = self.client[database]
            self.collection = self.database[collection]

        except Exception as e:
            raise NetworkSecurityError(e, sys)
        
    def csv_to_json(self, file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records=data.to_dict(orient="records")
            return records
        except Exception as e:
            raise NetworkSecurityError(e, sys)
        
    def insert_data_to_mongodb(self, records):
        try:
            self.collection.insert_many(records)
            return (len(records))
        except Exception as e:
            raise NetworkSecurityError(e, sys)

if __name__ == "__main__":
    networkObj = NetworkDataExtract(
        database="NetworkSecurity",
        collection="NetworkData")
    records = networkObj.csv_to_json(FILE_PATH)
    logging.info("Data converted successfully to json format")
    no_of_records = networkObj.insert_data_to_mongodb(records)
    logging.info(f"{no_of_records} records inserted successfully to MongoDB")