import os
import sys

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from src.exception import CustomException
from src.logger import logging

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)