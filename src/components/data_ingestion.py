import os
import sys
from typing import List

from src.exception import CustomException
from src.logger import logging

import pandas as pd

class DataIngestion:
    """
    Handles data loading and validation.
    """

    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads dataset from raw directory.

        Returns:
            pd.DataFrame: Loaded dataset.

        """
        try:
            logging.info("Loading data from raw data path.")

            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")

            df = pd.read_csv(self.raw_data_path)

            logging.info(f"Data loaded successfully with shape: {df.shape}")

            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def validate_data(df: pd.DataFrame, expected_columns: List[str]) -> None:
        """
        Validates the loaded data against expected columns.

        Args:
            df (pd.DataFrame): Loaded dataset.
            expected_columns (List[str]): List of expected column names.

        Raises:
            ValueError: If schema mismatch.
        """
        logging.info("Validating data schema.")

        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Schema mismatch! Missing columns: {missing_columns}")

        logging.info("Data schema validation passed.")

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Saves processed data to processed folder.

        Args:
            df (pd.DataFrame): Clean dataset.
        """

        try:
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)

            df.to_csv(self.processed_data_path, index=False, header=True)
            logging.info(f"Processed data saved successfully at {self.processed_data_path}.")
            
        except Exception as e:
            raise CustomException(e, sys)