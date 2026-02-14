import os
import sys

from sklearn.base import r2_score
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
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred)
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, params, models: dict) -> dict:
    """
    Evaluates multiple classification models and returns their performance metrics.
    """
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = params[list(models.keys())[i]]

            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
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
            X = self.preprocessor.transform(X)
            y_hat = self.model.predict(X)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)

