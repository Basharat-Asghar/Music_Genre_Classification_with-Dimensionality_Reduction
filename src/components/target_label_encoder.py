import os
import sys

import pandas as pd
import numpy as np 
import joblib

from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object, load_object

@dataclass
class TargetLabelEncoderConfig:
    label_encoder_obj_file_path: str = os.path.join("artifacts", "label_encoder.pkl")

class TargetLabelEncoder:
    """
    Handles Target Label Encoding operations.
    """

    def __init__(self):
        self.target_label_encoder_config = TargetLabelEncoderConfig()
        self.encoder = LabelEncoder()

    def fit_transform(self, y: pd.Series):
        try:
            logging.info("Starting target label encoding process.")

            # Fit and transform the target variable
            y_encoded = self.encoder.fit_transform(y)
            logging.info("Target label encoding completed successfully.")

            # Save the label encoder object
            save_object(
                file_path=self.target_label_encoder_config.label_encoder_obj_file_path,
                obj=self.encoder
            )
            logging.info("Label encoder object saved successfully.")

            return y_encoded

        except Exception as e:
            raise CustomException(e, sys)
        
    def inverse_transform(self, y_encoded: np.ndarray):
        try:
            logging.info("Starting inverse transformation of target labels.")

            # Load the label encoder object
            encoder = load_object(self.target_label_encoder_config.label_encoder_obj_file_path)

            # Inverse transform the encoded labels
            y_original = encoder.inverse_transform(y_encoded)
            logging.info("Inverse transformation of target labels completed successfully.")

            return y_original

        except Exception as e:
            raise CustomException(e, sys)

