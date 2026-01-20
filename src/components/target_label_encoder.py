import os
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class TargetLabelEncoderConfig:
    label_encoder_obj_file_path: str = os.path.join("artifacts", "label_encoder.pkl")

class TargetLabelEncoder:
    """
    Handles Target Label Encoding operations.
    """

    def __init__(self):
        self.target_label_encoder_config = TargetLabelEncoderConfig()

    def get_label_encoder(self):
        try:
            logging.info("Creating label encoder for target variable.")
            label_encoder = LabelEncoder()
            logging.info("Label encoder created successfully.")
            return label_encoder
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_target_label_encoding(self, train_arr: pd.DataFrame, test_arr: pd.DataFrame, target_column: str):
        try:
            logging.info("Starting target label encoding process.")

            label_encoder = self.get_label_encoder()
            # Fit label encoder on training target column
            train_arr[target_column] = label_encoder.fit_transform(train_arr[target_column])
            # Transform test target column
            test_arr[target_column] = label_encoder.transform(test_arr[target_column])

            logging.info("Target label encoding completed successfully.")

            # Save the label encoder object
            save_object(
                file_path=self.target_label_encoder_config.label_encoder_obj_file_path,
                obj=label_encoder
            )

            return (
                train_arr,
                test_arr,
                self.target_label_encoder_config.label_encoder_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)