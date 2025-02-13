import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    transformed_train_csv_path = os.path.join("artifacts", "train_transformed.csv")
    transformed_test_csv_path = os.path.join("artifacts", "test_transformed.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # External function for gender transformation
    def gender_transformer(self, df):
        df["Gender"] = df["Gender"].replace({"Male": 0, "Female": 1})
        return df

    def get_data_transformer(self):
        try:
            # Define columns for one hot encoding
            cols_one_hot_encoder = ["Geography"]

            # Create the preprocessor pipeline
            preprocessor = Pipeline(
                steps=[
                    (
                        "gender_transform",
                        FunctionTransformer(self.gender_transformer, validate=False),
                    ),
                    (
                        "one_hot_encoder",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "Geography_OHE",
                                    OneHotEncoder(
                                        handle_unknown="ignore", drop="first"
                                    ),
                                    cols_one_hot_encoder,
                                )
                            ],
                            remainder="passthrough",  # Keeps numerical features as they are
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),  # StandardScaler for numerical features
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Start preprocessor object
            preprocessor_object = self.get_data_transformer()

            target_column = "Exited"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            # not required for transformation
            # target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            # not required for transformation
            # target_feature_test_df = test_df[target_column]

            logging.info(f"Training columns: {input_feature_train_df.columns.tolist()}")
            logging.info(f"Target columns: {[target_column]}")

            # Fit the preprocessor on training data and transform it
            input_feature_train_array = preprocessor_object.fit_transform(
                input_feature_train_df
            )
            input_feature_test_array = preprocessor_object.transform(
                input_feature_test_df
            )

            # Convert arrays back to DataFrame, keeping only transformed data (no need for column names)
            train_df_transformed = pd.DataFrame(input_feature_train_array)
            test_df_transformed = pd.DataFrame(input_feature_test_array)

            # Save transformed data
            transformed_train_csv_path = (
                self.data_transformation_config.transformed_train_csv_path
            )
            transformed_test_csv_path = (
                self.data_transformation_config.transformed_test_csv_path
            )

            train_df_transformed.to_csv(transformed_train_csv_path, index=False)
            test_df_transformed.to_csv(transformed_test_csv_path, index=False)

            logging.info(
                f"Transformed train data saved at: {transformed_train_csv_path}"
            )
            logging.info(f"Transformed test data saved at: {transformed_test_csv_path}")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_object,
            )
            logging.info("Preprocessing object saved")

            return (
                transformed_train_csv_path,
                transformed_test_csv_path,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
