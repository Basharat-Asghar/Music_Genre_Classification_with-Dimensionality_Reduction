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
        
class CustomData:
    def __init__(self,
                 tempo: float,
                 dynamics_range: float,
                 vocal_presence: float,
                 percussion_strength: float,
                 string_instrument_detection: float,
                 electronic_element_presence: float,
                 rhythm_complexity: float,
                 drums_influence: float,
                 distorted_guitar: float,
                 metal_frequencies: float,
                 ambient_sound_influence: float,
                 instrumental_overlaps: float):
        self.tempo = tempo
        self.dynamics_range = dynamics_range
        self.vocal_presence = vocal_presence
        self.percussion_strength = percussion_strength
        self.string_instrument_detection = string_instrument_detection
        self.electronic_element_presence = electronic_element_presence
        self.rhythm_complexity = rhythm_complexity
        self.drums_influence = drums_influence
        self.distorted_guitar = distorted_guitar
        self.metal_frequencies = metal_frequencies
        self.ambient_sound_influence = ambient_sound_influence
        self.instrumental_overlaps = instrumental_overlaps

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "tempo": [self.tempo],
                "dynamics_range": [self.dynamics_range],
                "vocal_presence": [self.vocal_presence],
                "percussion_strength": [self.percussion_strength],
                "string_instrument_detection": [self.string_instrument_detection],
                "electronic_element_presence": [self.electronic_element_presence],
                "rhythm_complexity": [self.rhythm_complexity],
                "drums_influence": [self.drums_influence],
                "distorted_guitar": [self.distorted_guitar],
                "metal_frequencies": [self.metal_frequencies],
                "ambient_sound_influence": [self.ambient_sound_influence],
                "instrumental_overlaps": [self.instrumental_overlaps]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
