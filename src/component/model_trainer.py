import sys
from dataclasses import dataclass
import os
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_mnodel_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init_(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # all rows, ex-last column
                train_array[:, -1],  # all rows, last column
                test_array[:, :-1],
                test_array[:, -1],
            )
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)
