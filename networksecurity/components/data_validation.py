from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils import read_yaml_file, write_yaml_file
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
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {dataframe.columns}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_data_drif(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status=True
            report = {}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if is_same_dist.pvalue > threshold:
                    is_Found = False
                else:
                    is_Found = True
                    status=False
                report.update({column:{
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_Found
                }})

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            trained_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(trained_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # validate number of columns
            error_messages = []
            train_schema_status = self.validate_number_of_columns(train_df)
            if not train_schema_status:
                error_messages.append("Train dataframe does not have the required number of columns.\n")
                self.data_validation_config.valid_train_file_path=None
            test_schema_status = self.validate_number_of_columns(test_df)
            if not test_schema_status:
                self.data_validation_config.valid_test_file_path=None
                error_messages.append("Test dataframe does not have the required number of columns.\n")

            #checking data drift
            train_drift_status = self.detect_data_drif(base_df=train_df, current_df=test_df)
            if not train_drift_status:
                self.data_validation_config.valid_train_file_path=None
                error_messages.append("Data drift detected between train and test data.\n")
                
            test_drift_status = self.detect_data_drif(base_df=test_df, current_df=test_df)
            if not test_drift_status:
                self.data_validation_config.valid_test_file_path=None
                error_messages.append("Data drift detected in test data.\n")

            if  train_schema_status and train_drift_status:
                train_output_file_path =self.data_validation_config.valid_train_file_path
                os.makedirs(os.path.dirname(train_output_file_path), exist_ok=True)
                train_df.to_csv(train_output_file_path, index=False)
                self.data_validation_config.invalid_train_file_path=None
            else:
                train_output_file_path =self.data_validation_config.invalid_train_file_path
                os.makedirs(os.path.dirname(train_output_file_path), exist_ok=True)
                train_df.to_csv(train_output_file_path, index=False)
                self.data_validation_config.invalid_train_file_path=None

            if  test_schema_status and test_drift_status:
                test_output_file_path =self.data_validation_config.valid_test_file_path
                os.makedirs(os.path.dirname(test_output_file_path), exist_ok=True)
                test_df.to_csv(test_output_file_path, index=False)
                self.data_validation_config.invalid_test_file_path=None
            else:
                test_output_file_path =self.data_validation_config.invalid_test_file_path
                os.makedirs(os.path.dirname(test_output_file_path), exist_ok=True)
                test_df.to_csv(test_output_file_path, index=False)
                self.data_validation_config.invalid_test_file_path=None
            
            if len(error_messages) > 0:
                raise Exception(f"Data Validation Error: {'\n'.join(error_messages)}")

            return DataValidationArtifact(
                validation_status = train_schema_status and test_schema_status and train_drift_status and test_drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e