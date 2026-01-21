import os
import sys

import numpy as np
from sklearn.model_selection import cross_val_score

from src.exception import CustomException
from src.logger import logging

class ModelValidator:
    def __init__(self, model, X, y, cv=5, scoring='f1_macro'):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring

    def validate(self):
        try:
            logging.info("Starting model validation using cross-validation")
            scores = cross_val_score(
                self.model,
                self.X,
                self.y,
                cv=self.cv,
                scoring=self.scoring
            )

            logging.info(f"Cross-validation scores: {scores}")
            logging.info(f"Mean {self.scoring} score: {np.mean(scores)}")
            logging.info(f"Standard deviation of {self.scoring} score: {np.std(scores)}")

            return np.mean(scores), np.std(scores)

        except Exception as e:
            raise CustomException(e, sys)
