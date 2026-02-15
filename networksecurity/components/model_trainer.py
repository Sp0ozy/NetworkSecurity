import os 
import sys
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.utils.main_utils import save_object,  load_object, load_numpy_array_data
from networksecurity.utils.ml_utils import get_classification_score, evaluate_models, NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

import dagshub
dagshub.init(repo_owner='Sp0ozy', repo_name='NetworkSecurity', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config: ModelTrainerConfig = model_trainer_config
            self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "LogisticRegression": LogisticRegression(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier()
            }
            params = {
                "LogisticRegression": {
                    "solver": ["liblinear", "lbfgs"],
                },
                "KNeighborsClassifier": {
                    "n_neighbors": [3, 5, 9, 15, 25, 35],
                    # "weights": ["uniform", "distance"],
                    # "p": [1, 2],
                    # "leaf_size": [20, 30, 50]
                },
                "DecisionTreeClassifier": {
                    # "criterion": ["squared_error", "absolute_error", "friedman_mse"],
                    # "splitter": ["best"],
                    "max_depth": [None, 3, 5, 8, 12],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 5, 10],
                    "max_features": [None, "sqrt", "log2"]
                },
                "RandomForestClassifier": {
                    "n_estimators": [300, 600, 1000],
                    "max_depth": [None, 6, 10, 16],
                    "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 5],
                    # "max_features": ["sqrt", 0.5, 1.0],
                    # "bootstrap": [True]
                },
                "GradientBoostingClassifier": {
                    "learning_rate": [0.05, 0.1, 0.2],
                    "n_estimators": [200, 500, 1000],
                    # "max_depth": [2, 3, 4],
                    # "subsample": [0.7, 0.85, 1.0],
                    # "min_samples_leaf": [1, 3, 10],
                    # "max_features": [None, "sqrt"]
                },
                "AdaBoostClassifier": {
                    "n_estimators": [200, 500, 1000],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    # "loss": ["linear", "square", "exponential"]
                }
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  models=models, params=params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(f"Best found model on both training and testing dataset is {best_model_name}")

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            # Track the mlflow 

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.join(self.model_trainer_config.model_trainer_dir, best_model_name)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            logging.info("Loading transformed training and testing data")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            logging.info("Loaded transformed training and testing data successfully")

            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
