import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split

from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# configuration of Data Ingestion Component
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info(f"Data Ingestion component initialized with config: {self.data_ingestion_config}")
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    def export_colletion_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            logging.info(f"Connecting to MongoDB at {MONGO_DB_URL}")
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            logging.info(f"Accessing database: {database_name}, collection: {collection_name}")
            database = mongo_client[database_name]
            collection = database[collection_name]
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Data fetched from MongoDB, shape: {df.shape}")

            if '_id' in df.columns:
                df.drop('_id', axis=1, inplace=True)
                logging.info("Dropped '_id' column from data")
            
            df.replace('na', np.nan, inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    def export_data_into_feature_store(self,df:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Data exported to feature store at {feature_store_file_path}")
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
    
    def split_data_as_train_test(self, df:pd.DataFrame):
        try:
            train_set, test_set = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info(f"Data split into train and test sets with ratio {self.data_ingestion_config.train_test_split_ratio}")

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            logging.info(f"Train set saved at {self.data_ingestion_config.train_file_path}")

            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info(f"Test set saved at {self.data_ingestion_config.test_file_path}")

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_colletion_as_dataframe()
            logging.info("Data is successfully ingested into the DataFrame")
            self.export_data_into_feature_store(dataframe)
            logging.info(f"Data is successfully exported to feature store at {self.data_ingestion_config.feature_store_file_path}")
            self.split_data_as_train_test(dataframe)
            logging.info("Data is successfully split into train and test sets and saved to respective paths")
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            )
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e