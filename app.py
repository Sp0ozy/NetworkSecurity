import os
import sys

import certifi
from fastapi import FastAPI, File, Request, UploadFile
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

from networksecurity.utils.ml_utils import NetworkModel

ca=certifi.where()

from dotenv import load_dotenv   
load_dotenv() 

mongodb_urls = os.getenv("MONGO_DB_URL")

logging.info(f"Connecting to MongoDB (database={DATA_INGESTION_DATABASE_NAME}, collection={DATA_INGESTION_COLLECTION_NAME})")
client = pymongo.MongoClient(mongodb_urls, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]
logging.info("MongoDB connection established")

app = FastAPI()
orgins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=orgins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        logging.info("Received request to /train — starting training pipeline")
        train_pipeline = TrainPipeline(training_pipeline_config=TrainingPipelineConfig())
        train_pipeline.run_pipeline()
        logging.info("Training pipeline completed successfully via /train endpoint")
        return Response(content="Training successful!!", media_type="text/plain")
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file:UploadFile = File(...)):
    try:
        logging.info(f"Received prediction request for file: {file.filename}")
        df=pd.read_csv(file.file)
        logging.info(f"Uploaded CSV loaded with shape {df.shape}")
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        y_pred  = network_model.predict(df)
        df["predicted_column"] = y_pred
        logging.info(f"Prediction complete — {len(y_pred)} samples, saving output CSV")
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)
        table_html = df.to_html(classes = "table table-striped")

        return templates.TemplateResponse("table.html", {"request": request, "table_html": table_html})

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("Starting FastAPI app on 0.0.0.0:8000")
        app_run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        raise NetworkSecurityException(e, sys)