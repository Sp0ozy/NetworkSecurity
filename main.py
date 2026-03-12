from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

import sys

if __name__ == "__main__":
    try:
        logging.info("========== Training run started ==========")
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Starting data ingestion process")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed successfully. Data Ingestion Artifact: {data_ingestion_artifact}")

        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                         data_validation_config=data_validation_config)
        logging.info("Starting data validation process")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation completed successfully. Data Validation Artifact: {data_validation_artifact}")

        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                    data_transformation_config=data_transformation_config)
        logging.info("Starting data transformation process")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data Transformation completed successfully. Data Transformation Artifact: {data_transformation_artifact}")

        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_transformation_artifact=data_transformation_artifact)
        logging.info("Starting model trainer process")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model Trainer completed successfully. Model Trainer Artifact: {model_trainer_artifact}")
        logging.info("========== Training run completed ==========")

    except Exception as e:
        raise NetworkSecurityException(e,sys) from e