import os
import sys

from networksecurity.logging.logger import logging
from networksecurity.exceptions.exception import NetworkSecurityException

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.constants.training_pipeline import (
    TRAINING_BUCKET_NAME
)
from networksecurity.cloud.s3_sync import S3Sync

class TrainPipeline:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config
            self.s3_sync = S3Sync()
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Data Ingestion config created successfully")
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            logging.info("Starting data ingestion process")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed successfully. Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Data Validation config created successfully")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            logging.info("Starting data validation process")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed successfully. Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Data Transformation config created successfully")
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,    
                                                      data_transformation_config=data_transformation_config)
            logging.info("Starting data transformation process")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed successfully. Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                         data_transformation_artifact=data_transformation_artifact)
            logging.info("Starting model trainer process")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model Trainer completed successfully. Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    # local artifact is going to s3 bucket
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            logging.info(f"Syncing artifact dir to S3: {self.training_pipeline_config.artifact_dir} -> {aws_bucket_url}")
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
            logging.info("Artifact dir sync to S3 completed")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # local final is going to s3 bucket
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            logging.info(f"Syncing final model dir to S3: {self.training_pipeline_config.model_dir} -> {aws_bucket_url}")
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir, aws_bucket_url=aws_bucket_url)
            logging.info("Final model dir sync to S3 completed")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def run_pipeline(self):
        try:
            logging.info("========== Training pipeline started ==========")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
            logging.info("========== Training pipeline completed ==========")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

