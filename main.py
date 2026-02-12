from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig
from networksecurity.exceptions.exception import NetworkSecurityException

import sys

if __name__ == "__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        print("Starting data ingestion process")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(f"Data Ingestion completed successfully. Data Ingestion Artifact: {data_ingestion_artifact}")

        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                         data_validation_config=data_validation_config)
        print("Starting data validation process")
        data_validation_artifact = data_validation.initiate_data_validation()
        print(f"Data Validation completed successfully. Data Validation Artifact: {data_validation_artifact}")
        
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                    data_transformation_config=data_transformation_config)
        print("Starting data transformation process")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(f"Data Transformation completed successfully. Data Transformation Artifact: {data_transformation_artifact}")
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e