import os
import sys

import certifi
from fastapi import FastAPI
import pymongo

from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.logging.logger import logging
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainPipeline
from networksecurity.utils.main_utils import load_object

from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME

from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

import pandas as pd

ca=certifi.where()

from dotenv import load_dotenv   
load_dotenv() 

mongodb_urls = os.getenv("MONGODB_URI")

client = pymongo.MongoClient(mongodb_urls, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
orgins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=orgins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline( training_pipeline_config=TrainingPipelineConfig())
        train_pipeline.run_pipeline()
        return Response(content="Training successful!!", media_type="text/plain")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    try:
        app_run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        raise NetworkSecurityException(e, sys)