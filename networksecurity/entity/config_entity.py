from datetime import datetime
import os
from networksecurity.constants import training_pipeline

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')):
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = f"{training_pipeline.PIPELINE_NAME}{timestamp}"
        self.artifact_dir = training_pipeline.ARTIFACT_DIR
        self.timestamp = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path:str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME)
        self.train_file_path:str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME)
        self.test_file_path:str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME)
        self.train_test_split_ratio:float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name:str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name:str = training_pipeline.DATA_INGESTION_DATABASE_NAME
        
        
        