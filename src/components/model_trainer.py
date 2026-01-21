import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging

class ModelTrainer:
    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        try:
            logging.info("Splitting data into training and testing sets.")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.y
            )
            logging.info("Data splitting completed.")

            return (
                X_train, X_test, y_train, y_test
            )

        except Exception as e:
            raise CustomException(e, sys)
        
    def train_log_cls(self, max_iter: int = 1000):
        try:
            logging.info("Training Logistic Regression model.")
            model = LogisticRegression(
                max_iter=max_iter,
                random_state=self.random_state
            )

            model.fit(self.X_train, self.y_train)
            logging.info("Logistic Regression model training completed.")

            return model
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def train_knn_cls(self, n_neighbors: int = 5):
        try:
            logging.info("Training K-Nearest Neighbors Classifier.")
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors
            )
            model.fit(self.X_train, self.y_train)
            logging.info("K-Nearest Neighbors Classifier training completed.")

            return model
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def train_svc_cls(self, kernel: str = 'rbf', C: float = 1.0):
        try:
            logging.info("Training Support Vector Classifier.")
            model = SVC(
                kernel=kernel,
                C=C,
                random_state=self.random_state,
                probability=True
            )
            model.fit(self.X_train, self.y_train)
            logging.info("Support Vector Classifier training completed.")

            return model

        except Exception as e:
            raise CustomException(e, sys)
        
    def train_models(self):
        try:
            logging.info("Starting model training process.")
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

            models = {
                "Logistic Regression": self.train_log_cls(max_iter=1000),
                "K-Nearest Neighbors": self.train_knn_cls(n_neighbors=5),
                "Support Vector Classifier": self.train_svc_cls(kernel='rbf', C=1.0)
            }

            logging.info("Model training process completed.")

            return models, self.X_test, self.y_test
        
        except Exception as e:
            raise CustomException(e, sys)
        