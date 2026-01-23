import os
import sys

import pandas as pd

from src.utils import load_object
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

@dataclass
class PredictPipelineConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")
    target_label_encoder_obj_path = os.path.join("artifacts", "label_encoder.pkl")
    pca_obj_path = os.path.join("artifacts", "pca_model.pkl")
    model_obj_path = os.path.join("artifacts", "tuned_model.pkl")

class PredictPipeline:
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict(self, input_df: pd.DataFrame):
        try:
            logging.info("Started loading objects to predict genre.")
            preprocessor = load_object(self.predict_pipeline_config.preprocessor_obj_path)
            target_label_encoder = load_object(self.predict_pipeline_config.target_label_encoder_obj_path)
            pca_model = load_object(self.predict_pipeline_config.pca_obj_path)
            model = load_object(self.predict_pipeline_config.model_obj_path)
            logging.info("All objects loaded successfully.")

            logging.info("Applying same processing as training.")
            X_scaled = preprocessor.transform(input_df)
            X_pca = pca_model.transform(X_scaled)
            logging.info("Processing is done.")

            logging.info("Predicting Music Genre.")
            pred = model.predict(X_pca)
            genre = target_label_encoder.inverse_transform(pred)
            logging.info(f"Predict Music Genre is: {genre}")

            return genre

        except Exception as e:
            raise CustomException(e, sys)