import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging  
from sklearn.metrics import precision_score, recall_score, f1_score


from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME



def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Computes classification metrics and returns them in a ClassificationMetricArtifact object.
    """
    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        logging.info(f"Classification metrics — f1={f1:.4f}  precision={precision:.4f}  recall={recall:.4f}")
        return ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, params, models: dict) -> dict:
    """
    Evaluates multiple classification models and returns their performance metrics.
    """
    try:
        report={}
        model_names = list(models.keys())
        for i in range(len(model_names)):
            model_name = model_names[i]
            model = list(models.values())[i]
            model_params = params[model_name]

            logging.info(f"Evaluating model: {model_name}  (param grid: {model_params})")
            gs = GridSearchCV(model, model_params, cv=3)
            gs.fit(X_train, y_train)
            logging.info(f"Best params for {model_name}: {gs.best_params_}")

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            logging.info(f"{model_name} — train R2={train_model_score:.4f}  |  test R2={test_model_score:.4f}")

            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)

class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, X):
        try:
            logging.info(f"Running prediction on input with shape {X.shape}")
            X = self.preprocessor.transform(X)
            y_hat = self.model.predict(X)
            logging.info(f"Prediction complete — {len(y_hat)} samples predicted")
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)

