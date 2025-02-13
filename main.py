import sys
from src.exception import CustomException
from src.component.data_ingestion import DataIngestion, DataIngestionConfig
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        data_ingestion_obj = DataIngestion()
        data_ingestion_config = DataIngestionConfig()
        df = data_ingestion_obj.initiate_data_ingestion()
        data_transformation_obj = DataTransformation()
        (
            train_array,
            test_array,
            _,
            _,
            _,
        ) = data_transformation_obj.initiate_data_transformation(
            data_ingestion_config.train_data_path, data_ingestion_config.test_data_path
        )
        model_trainer_config = ModelTrainer()
        model_trainer_config.initiate_model_trainer(train_array, test_array)

    except Exception as e:
        raise CustomException(e, sys)
