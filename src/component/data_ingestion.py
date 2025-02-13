import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.data_path: str = os.path.join("data", "Churn_Modelling.csv")
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Initiates the data ingestion process.

        This method reads the dataset from notebooks/data/, creates a directory for storing the train,
        test, and raw data, performs a train-test split, and saves the data to the respective files.

        Return:
            A tuple containing the file paths for the train and test data
        """
        logging.info("Initializing data ingestion component")
        try:
            df = pd.read_csv(self.data_path)
            logging.info(f"Shape of dataframe: {df.shape}")
            # remove unnecessary columns
            df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], axis=1)

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(
                self.data_ingestion_config.raw_data_path, index=False, header=True
            )

            logging.info(
                f"Dataframe shape: {df.shape}, \nDataframe columns: {list(df.columns)}"
            )

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            train_set.to_csv(
                self.data_ingestion_config.train_data_path, index=False, header=True
            )

            logging.info(f"Train dataframe shape: {train_set.shape}")
            test_set.to_csv(
                self.data_ingestion_config.test_data_path, index=False, header=True
            )

            logging.info(f"Test dataframe shape: {test_set.shape}.")
            logging.info("Data ingestion process completed successfully.")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
