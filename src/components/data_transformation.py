import os
import sys

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    #label_encoder_obj_file_path: str = os.path.join("artifacts", "label_encoder.pkl")

class DataTransformation:
    """
    Handles Data Transformation operations.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, df: pd.DataFrame):

        try:
            logging.info("Creating data transformer object.")

            # Identify categorical and numerical columns
            num_features = df.select_dtypes(exclude=['object']).columns.tolist()
            logging.info(f"Numerical features: {num_features}")

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_features)
                ]
            )
            logging.info("Data transformer object created successfully.")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        


    def fit_transform(self, X: pd.DataFrame):
        try:
            logging.info("Starting data transformation process.")

            # Get the data transformer object
            preprocessor = self.get_data_transformer_object(X)

            # Fit and transform the data
            X_scaled = preprocessor.fit_transform(X)
            logging.info("Data transformation completed.")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor object saved successfully.")

            return (
                pd.DataFrame(X_scaled, columns=X.columns),
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

    def transform(self, X: pd.DataFrame):
        try:
            logging.info("Starting data transformation for new data.")

            # Load the preprocessor object
            preprocessor = load_object(self.data_transformation_config.preprocessor_obj_file_path)

            # Transform the data
            X_scaled = preprocessor.transform(X)
            logging.info("Data transformation for new data completed.")

            return pd.DataFrame(X_scaled, columns=X.columns)

        except Exception as e:
            raise CustomException(e, sys)
        