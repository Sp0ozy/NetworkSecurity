import os
import sys
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
