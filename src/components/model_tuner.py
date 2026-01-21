import os
import sys

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

from src.exception import CustomException
from src.logger import logging

class HyperParameterTuner:
    def __init__(self, model, X, y, param_dist, iter=50, scoring='f1_macro', cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.param_dist = param_dist
        self.iter = iter
        self.scoring = scoring
        self.cv = cv

    def tuner(self):
        try:
            logging.info("Starting HyperParameter Tuning.")
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_dist,
                n_iter=self.iter,
                scoring=self.scoring,
                cv=self.cv,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            random_search.fit(self.X, self.y)

            logging.info("HyperParameter Tuning Completed.")
            logging.info(f"Best Parameters: {random_search.best_params_}")
            logging.info(f"Best CV f1_macro: {random_search.best_score_}")

            return random_search.best_estimator_

        except Exception as e:
            raise CustomException(e, sys)