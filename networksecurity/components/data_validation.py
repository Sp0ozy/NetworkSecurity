from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils import read_yaml_file
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH

from scipy.stats import ks_2samp
import pandas as pd
import os,sys

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            trained_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(trained_file_path)
            test_df = DataValidation.read_data(test_file_path)





        except Exception as e:
            raise NetworkSecurityException(e, sys) from e