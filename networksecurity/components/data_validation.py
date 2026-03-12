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
            logging.info(f"DataValidation initialized with schema from {SCHEMA_FILE_PATH}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from {file_path}")
            df = pd.read_csv(file_path)
            logging.info(f"Loaded dataframe with shape {df.shape} from {file_path}")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {dataframe.columns}")
            if len(dataframe.columns) == number_of_columns:
                logging.info(f"Column count validation passed: {len(dataframe.columns)} columns found")
                return True
            logging.warning(f"Column count validation FAILED: expected {number_of_columns} columns, got {len(dataframe.columns)}")
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_data_drif(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            logging.info(f"Running KS drift detection on {len(base_df.columns)} columns (threshold={threshold:.2f})")
            status=True
            report = {}
            drifted_columns = []
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if is_same_dist.pvalue > threshold:
                    is_Found = False
                else:
                    is_Found = True
                    status=False
                    drifted_columns.append(column)
                report.update({column:{
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_Found
                }})

            if drifted_columns:
                logging.warning(f"Drift detected in {len(drifted_columns)} column(s): {drifted_columns}")
            else:
                logging.info("No drift detected in any column")

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            logging.info(f"Drift report saved to {drift_report_file_path}")
            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            trained_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(trained_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # validate number of columns
            error_messages = []
            train_schema_status = self.validate_number_of_columns(train_df)
            if not train_schema_status:
                logging.error("Train dataframe column count mismatch")
                error_messages.append("Train dataframe does not have the required number of columns.\n")
                self.data_validation_config.valid_train_file_path=None
            test_schema_status = self.validate_number_of_columns(test_df)
            if not test_schema_status:
                logging.error("Test dataframe column count mismatch")
                self.data_validation_config.valid_test_file_path=None
                error_messages.append("Test dataframe does not have the required number of columns.\n")

            #checking data drift
            logging.info("Checking for data drift between train and test sets")
            train_drift_status = self.detect_data_drif(base_df=train_df, current_df=test_df)
            if not train_drift_status:
                logging.warning("Data drift detected between train and test data")
                self.data_validation_config.valid_train_file_path=None
                error_messages.append("Data drift detected between train and test data.\n")

            logging.info("Checking for internal drift within test set")
            test_drift_status = self.detect_data_drif(base_df=test_df, current_df=test_df)
            if not test_drift_status:
                self.data_validation_config.valid_test_file_path=None
                error_messages.append("Data drift detected in test data.\n")

            if  train_schema_status and train_drift_status:
                train_output_file_path =self.data_validation_config.valid_train_file_path
                os.makedirs(os.path.dirname(train_output_file_path), exist_ok=True)
                train_df.to_csv(train_output_file_path, index=False)
                logging.info(f"Valid train data saved to {train_output_file_path}")
                self.data_validation_config.invalid_train_file_path=None
            else:
                train_output_file_path =self.data_validation_config.invalid_train_file_path
                os.makedirs(os.path.dirname(train_output_file_path), exist_ok=True)
                train_df.to_csv(train_output_file_path, index=False)
                logging.warning(f"Train data failed validation; saved to invalid path: {train_output_file_path}")
                self.data_validation_config.invalid_train_file_path=None

            if  test_schema_status and test_drift_status:
                test_output_file_path =self.data_validation_config.valid_test_file_path
                os.makedirs(os.path.dirname(test_output_file_path), exist_ok=True)
                test_df.to_csv(test_output_file_path, index=False)
                logging.info(f"Valid test data saved to {test_output_file_path}")
                self.data_validation_config.invalid_test_file_path=None
            else:
                test_output_file_path =self.data_validation_config.invalid_test_file_path
                os.makedirs(os.path.dirname(test_output_file_path), exist_ok=True)
                test_df.to_csv(test_output_file_path, index=False)
                logging.warning(f"Test data failed validation; saved to invalid path: {test_output_file_path}")
                self.data_validation_config.invalid_test_file_path=None

            if len(error_messages) > 0:
                error_msg = '\n'.join(error_messages)
                logging.error(f"Data validation completed with errors: {error_msg}")
                raise Exception(f"Data Validation Error: {error_msg}")

            overall_status = train_schema_status and test_schema_status and train_drift_status and test_drift_status
            logging.info(f"Data validation completed. Overall status: {overall_status}")
            return DataValidationArtifact(
                validation_status=overall_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e