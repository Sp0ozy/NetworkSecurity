import os 
import sys
from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.utils.main_utils import save_object,  load_object, load_numpy_array_data
from networksecurity.utils.ml_utils import get_classification_score, evaluate_models, NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import mlflow

import dagshub

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config: ModelTrainerConfig = model_trainer_config
            self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classificatio_metric):
        try:
            logging.info("Initializing DagShub for MLflow tracking")
            dagshub.init(repo_owner='Sp0ozy', repo_name='NetworkSecurity', mlflow=True)
        except Exception as e:
            logging.warning(f"DagHub initialization failed: {e}. MLflow tracking may not work.")

        logging.info("Starting MLflow run to log metrics and model")
        with mlflow.start_run():
            mlflow.log_param("f1_score", classificatio_metric.f1_score)
            mlflow.log_param("precision_score", classificatio_metric.precision_score)
            mlflow.log_param("recall_score", classificatio_metric.recall_score)
            mlflow.sklearn.log_model(best_model, "model")
        logging.info(f"MLflow run completed — f1={classificatio_metric.f1_score:.4f}  precision={classificatio_metric.precision_score:.4f}  recall={classificatio_metric.recall_score:.4f}")
        
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
                    # "max_features": [None, "sqrt", "log2"]
                },
                "RandomForestClassifier": {
                    # "n_estimators": [300, 600, 1000],
                    "max_depth": [None, 6, 10, 16],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 5],
                    # "max_features": ["sqrt", 0.5, 1.0],
                    # "bootstrap": [True]
                },
                "GradientBoostingClassifier": {
                    "learning_rate": [0.05, 0.1, 0.2],
                    # "n_estimators": [200, 500, 1000],
                    # "max_depth": [2, 3, 4],
                    # "subsample": [0.7, 0.85, 1.0],
                    # "min_samples_leaf": [1, 3, 10],
                    # "max_features": [None, "sqrt"]
                },
                "AdaBoostClassifier": {
                    # "n_estimators": [200, 500, 1000],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    # "loss": ["linear", "square", "exponential"]
                }
            }
            logging.info(f"Starting model evaluation — X_train: {X_train.shape}  |  X_test: {X_test.shape}")
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  models=models, params=params)

            logging.info(f"Model evaluation scores: {model_report}")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(f"Best model: {best_model_name}  (test f1={best_model_score:.4f})")

            if best_model_score < self.model_trainer_config.expected_score:
                logging.warning(f"Best model score {best_model_score:.4f} is below expected threshold {self.model_trainer_config.expected_score:.4f}")

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            logging.info(f"Train metrics — f1={classification_train_metric.f1_score:.4f}  precision={classification_train_metric.precision_score:.4f}  recall={classification_train_metric.recall_score:.4f}")

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            logging.info(f"Test metrics  — f1={classification_test_metric.f1_score:.4f}  precision={classification_test_metric.precision_score:.4f}  recall={classification_test_metric.recall_score:.4f}")

            # Track the mlflow
            self.track_mlflow(best_model, classification_test_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info(f"Loaded preprocessor from {self.data_transformation_artifact.transformed_object_file_path}")

            model_dir_path = os.path.join(self.model_trainer_config.model_trainer_dir, best_model_name)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            logging.info(f"NetworkModel saved to {self.model_trainer_config.trained_model_file_path}")

            save_object("final_model/model.pkl", best_model)
            logging.info("Best model also saved to final_model/model.pkl")

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

            logging.info(f"Loading transformed train array from {train_file_path}")
            train_arr = load_numpy_array_data(train_file_path)
            logging.info(f"Loading transformed test array from {test_file_path}")
            test_arr = load_numpy_array_data(test_file_path)
            logging.info(f"Arrays loaded — train: {train_arr.shape}  |  test: {test_arr.shape}")

            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]
            logging.info(f"Feature/target split — X_train: {X_train.shape}  y_train: {y_train.shape}  |  X_test: {X_test.shape}  y_test: {y_test.shape}")

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            logging.info("Model training completed successfully")

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
