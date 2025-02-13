import sys
from src.exception import CustomException
from src.component.data_ingestion import DataIngestion, DataIngestionConfig
from src.component.data_transformation import DataTransformation


if __name__ == "__main__":
    try:
        data_ingestion_obj = DataIngestion()
        data_ingestion_config = DataIngestionConfig()
        df = data_ingestion_obj.initiate_data_ingestion()
        data_transformation_obj = DataTransformation()
        df = data_transformation_obj.initiate_data_transformation(
            data_ingestion_config.train_data_path, data_ingestion_config.test_data_path
        )
    except Exception as e:
        raise CustomException(e, sys)
