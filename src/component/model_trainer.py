import sys
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.h5")
    tensorboard_log_dir: str = os.path.join("artifacts", "tensor_logs")
    loss_images_dir: str = os.path.join("artifacts", "loss_images")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_ann_model(self, X_train, y_train, X_test, y_test):
        try:
            # Initialize ANN model
            model = Sequential(
                [
                    Dense(
                        64, activation="relu", input_shape=(X_train.shape[1],)
                    ),  # HL 1
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.3),
                    Dense(32, activation="relu"),  # HL 2
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.3),
                    Dense(16, activation="relu"),  # HL 3
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.3),
                    Dense(1, activation="sigmoid"),  # Output layer
                ]
            )

            # Compile the model
            opt = tf.keras.optimizers.Adam(learning_rate=0.001)
            loss = tf.keras.losses.BinaryCrossentropy()
            model.compile(
                optimizer=opt,  # Adam optimizer
                loss=loss,  # Binary Crossentropy loss
                metrics=["accuracy"],
            )

            # Ensure the log and loss images directories exist
            os.makedirs(self.model_trainer_config.tensorboard_log_dir, exist_ok=True)
            os.makedirs(self.model_trainer_config.loss_images_dir, exist_ok=True)

            # Setup tensorboard callback
            tensorflow_callback = TensorBoard(
                log_dir=self.model_trainer_config.tensorboard_log_dir, histogram_freq=1
            )

            # Setup early stopping callback
            early_stopping_callback = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            # Train the model
            history = model.fit(
                X_train,
                y_train,
                epochs=100,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping_callback, tensorflow_callback],
            )

            # Plot and save the loss graph
            self.plot_loss(history)

            logging.info("Model training completed successfully.")
            return model

        except Exception as e:
            raise CustomException(e, sys)

    def plot_loss(self, history):
        try:
            # Create a plot for training and validation loss
            plt.plot(history.history["loss"], label="Training Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()

            # Save the plot
            loss_plot_path = os.path.join(
                self.model_trainer_config.loss_images_dir, "loss_plot.png"
            )
            plt.savefig(loss_plot_path)
            plt.close()
            logging.info(f"Loss plot saved at: {loss_plot_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All rows, excluding last column (features)
                train_array[:, -1],  # Last column as target (labels)
                test_array[:, :-1],  # All rows, excluding last column (features)
                test_array[:, -1],  # Last column as target (labels)
            )

            logging.info(f"Shape of X_train: {X_train.shape}")
            logging.info(f"Shape of y_train: {y_train.shape}")
            logging.info(f"Shape of X_test: {X_test.shape}")
            logging.info(f"Shape of y_test: {y_test.shape}")

            # Initialize and train the model
            model = self.initiate_ann_model(X_train, y_train, X_test, y_test)

            # Save the trained model
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(
                f"Model saved at: {self.model_trainer_config.trained_model_file_path}"
            )

            return model

        except Exception as e:
            raise CustomException(e, sys)
