# Network Security — ML Phishing/Attack Classifier

An end-to-end machine learning project that classifies network traffic records as malicious or benign. Built as a learning project to practice the full ML lifecycle: data ingestion, validation, transformation, model training, experiment tracking, and serving predictions via a web API.

---

## What It Does

Takes a CSV of network traffic features, runs it through a trained classifier, and returns a prediction for each record — flagging potentially malicious connections. You can trigger training or get predictions through a simple web interface.

---

## ML Approach & Architecture

The pipeline has four stages:

1. **Data Ingestion** — pulls raw data from a MongoDB collection and splits it into train/test sets
2. **Data Validation** — checks column counts and runs KS drift detection (scipy) to compare train vs test distributions
3. **Data Transformation** — imputes missing values with `KNNImputer`, binarizes the target column
4. **Model Trainer** — runs `GridSearchCV` across six classifiers and picks the best by F1 score:
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - AdaBoost

The best model is wrapped with its preprocessor into a `NetworkModel` object and saved. Metrics (F1, precision, recall) are logged to MLflow via DagShub.

---

## Tech Stack

| Area | Libraries |
|---|---|
| ML | scikit-learn, numpy, pandas, scipy |
| Experiment tracking | MLflow, DagShub |
| Web API | FastAPI, Uvicorn, Jinja2 |
| Database | MongoDB (pymongo) |
| Cloud storage | AWS S3 (boto3) |
| Config & utils | PyYAML, python-dotenv, certifi |

---

## Project Structure

```
networksecurity/
  components/         # Pipeline stages: ingestion, validation, transformation, trainer
  pipeline/           # training_pipeline.py — orchestrates all stages
  entity/             # Config and artifact dataclasses
  utils/              # main_utils.py (file I/O), ml_utils.py (metrics, evaluation, NetworkModel)
  constants/          # Training pipeline constants (paths, thresholds, DB names)
  logging/            # Centralized logger writing to logs/
  exceptions/         # Custom NetworkSecurityException
  cloud/              # S3 sync helper
app.py                # FastAPI app — /train and /predict endpoints
main.py               # Script entrypoint for running the pipeline directly
templates/            # Jinja2 HTML template for prediction output table
data_schema/          # schema.yaml — expected column definitions
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set environment variables** — create a `.env` file:
```
MONGO_DB_URL=your_mongodb_connection_string
```

**3. Run the web app**
```bash
python app.py
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs).

- `GET /train` — runs the full training pipeline
- `POST /predict` — upload a CSV file, returns a table with predictions

**Or run training directly:**
```bash
python main.py
```

---

## Deployment

The project includes AWS S3 sync for storing artifacts and a Docker/CI setup for cloud deployment. **Note: the AWS EC2 instance is currently suspended, so live deployment is not available.**

---

## What I Learned

- How to structure a real ML project into modular, reusable components instead of one big notebook
- How the full training pipeline works end-to-end — from raw data to a served model
- Data validation and drift detection before training
- Hyperparameter tuning with `GridSearchCV`
- Experiment tracking with MLflow and DagShub
- Serving ML models via a REST API with FastAPI
- Centralized logging across a multi-file Python project
- Working with MongoDB as a data source

---

## Current Limitations & What Could Be Improved

This is a learning project and there are a few areas that could be stronger:

- **No input validation on uploaded CSVs** — if the file has missing or unexpected columns, the app crashes with a raw error instead of a helpful message
- **Hyperparameter search is limited** — most of the `GridSearchCV` param grids are commented out to keep training fast; a proper grid search would likely improve model performance
- **No tests** — there are no unit or integration tests for any of the pipeline components
- **No model versioning** — every training run overwrites `final_model/model.pkl`; a real system would version models and allow rollback

---

## Tools & AI Assistance

I used **Claude** (by Anthropic) throughout this project as a coding assistant — for debugging issues, adding structured logging across the codebase, fixing bugs (wrong evaluation metric, broken HTML template, encoding errors on Windows), and reviewing code for inconsistencies. This README was written with Claude's help.
