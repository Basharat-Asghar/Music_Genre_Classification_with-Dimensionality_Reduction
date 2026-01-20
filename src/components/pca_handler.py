import os
import sys

import pandas as pd
import numpy as np
import joblib

from sklearn.decomposition import PCA
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object
from dataclasses import dataclass

@dataclass
class PCAHandlerConfig:
    pca_model_path: str = os.path.join("artifacts", "pca_model.pkl")

class PCAHandler:
    def __init__(self, n_components: float = 0.85):
        """
        Args:
            n_components (float or int):
                - float (0-1): explained variance ratio to retain
                - int: number of components
        """

        self.pca_handler_config = PCAHandlerConfig()
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit_transform(self, X: pd.DataFrame):
        try:
            logging.info("Fitting and transforming data using PCA.")
            X_pca = self.pca.fit_transform(X)
            logging.info(f"PCA fitting and transformation completed. Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.2f}")

            # Save the PCA model
            save_object(
                file_path=self.pca_handler_config.pca_model_path,
                obj=self.pca
            )
            logging.info(f"PCA model saved at {self.pca_handler_config.pca_model_path}")

            return pd.DataFrame(
                X_pca,
                columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def transform(self, X: pd.DataFrame):
        try:
            logging.info("Transforming data using existing PCA model.")
            pca_model = load_object(self.pca_handler_config.pca_model_path)
            X_pca = pca_model.transform(X)
            logging.info("Data transformation using PCA completed.")

            return pd.DataFrame(
                X_pca,
                columns=[f"PC{i+1}"for i in range(pca_model.n_components_)]
            )
        except Exception as e:
            raise CustomException(e, sys)