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
            logging.info("Evaluating model performance.")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            logging.info("Model evaluation completed.")

            logging.info(f"Model: {model.__class__.__name__}")
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Macro F1 Score: {macro_f1}")
            logging.info(f"Classification Report:\n{report}")
            logging.info(f"Confusion Matrix:\n{cm}")

            return (
                accuracy,
                macro_f1,
                report,
                cm
            )
        except Exception as e:
            raise CustomException(e, sys)