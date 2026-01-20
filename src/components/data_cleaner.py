import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging

class DataCleaner:
    """
    Handles Data Cleaning operations.
    """

    def __init__(self):
        pass

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from the DataFrame.

        Args:
            df (pd.DataFrame): Input dataset.
        """
        try:
            logging.info("removing duplicate rows from the dataset.")
            df = df.drop_duplicates().reset_index(drop=True)
            logging.info(f"Duplicates removed. New shape: {df.shape}")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handles missing values in the DataFrame.

        Args:
            df (pd.DataFrame): Input dataset.
            strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop').

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        try:
            logging.info(f"Handling missing values using strategy: {strategy}")

            if strategy == 'mean':
                df = df.fillna(df.mean())
            elif strategy == 'median':
                df = df.fillna(df.median())
            elif strategy == 'mode':
                df = df.fillna(df.mode().iloc[0])
            elif strategy == 'drop':
                df = df.dropna().reset_index(drop=True)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            logging.info(f"Missing values handled successfully. New shape: {df.shape}")

            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names to lowercase and replaces spaces with underscores.

        Args:
            df (pd.DataFrame): Input dataset.

        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        """
        try:
            logging.info("Standardizing column names.")
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            logging.info(f"Column names standardized: {df.columns.tolist()}")
            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    