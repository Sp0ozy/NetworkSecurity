from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from networksecurity.exceptions.exception import NetworkSecurityError
import sys

if __name__ == "__main__":
    try:
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        print("Starting data ingestion process")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(f"Data Ingestion completed successfully. Data Ingestion Artifact: {data_ingestion_artifact}")
    except Exception as e:
        raise NetworkSecurityError(e,sys) from e