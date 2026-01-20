import os
import sys

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

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
        
    '''
    def get_label_encoder(self):
        try:
            logging.info("Creating label encoder for target variable.")

            label_encoder = LabelEncoder()

            logging.info("Label encoder created successfully.")

            return label_encoder

        except Exception as e:
            raise CustomException(e, sys)
    '''


    def initiate_data_transformation(self, df: pd.DataFrame):
        try:
            
            logging.info("Initiating data transformation.")

            preprocessor = self.get_data_transformer_object(df)

            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )
            logging.info("Data split into train and test sets.")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Separate features and target
            target_column_name = "genre"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Fit and transform the training data, transform the test data
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Data transformation completed.")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor object saved successfully.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
    '''
    def initiate_label_encoding(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logging.info("Initiating label encoding for target variable.")

            label_encoder = self.get_label_encoder()

            # Separate features and target
            target_column_name = "genre"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Fit and transform the training target, transform the test target
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info("Label encoding completed.")

            train_arr = np.c_[input_feature_train_df, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_df, target_feature_test_arr]

            # Save the label encoder object
            save_object(
                file_path=self.data_transformation_config.label_encoder_obj_file_path,
                obj=label_encoder
            )
            logging.info("Label encoder object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.label_encoder_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
    '''