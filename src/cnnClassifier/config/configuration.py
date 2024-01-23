import os
from cnnClassifier.constants import *
import tensorflow as tf
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, TrainingConfig
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params
        train_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(self.config.data_ingestion.unzip_dir, "Vegetable Images", "train"), shuffle=True)
        val_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(self.config.data_ingestion.unzip_dir, "Vegetable Images", "validation"), shuffle=True)
        test_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(self.config.data_ingestion.unzip_dir, "Vegetable Images", "test"), shuffle=True)
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            train_ds=list(train_ds),
            val_ds = list(val_ds),
            test_ds = list(test_ds),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            height=params.height,
            width=params.width,
            num_classes=params.CLASSES
        )

        return training_config
      