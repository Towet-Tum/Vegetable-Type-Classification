from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_training import TrainingPipeline



STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training Pipeline"
try:
    logger.info(f">>>>>>> The {STAGE_NAME} has started >>>>>>>>>>>>")
    training = TrainingPipeline()
    training.main()
    logger.info(f">>>>>>>>>>> The {STAGE_NAME} has completed successfully >>>>>>>>>>")
except Exception as e:
    raise e

