from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    train_ds: Path
    val_ds: Path
    test_ds: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    height: int
    width: int
    num_classes: int