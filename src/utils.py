import os
import sys

import pandas as pd
import numpy as np
import joblib

from src.exception import CustomException

def save_object(file_path: str, obj: object):
    """
    Saves a Python object to a file using joblib.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj (object): The Python object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path:str):
    """
    Loads a Python object from a file using joblib.

    Args:
        file_path (str): Path to the file from which the object will be loaded.

    Returns:
        object: The loaded Python object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)