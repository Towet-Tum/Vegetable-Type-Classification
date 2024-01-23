from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.training import Training

STAGE_NAME = "Training Pipeline"
class TrainingPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> The {STAGE_NAME} has started >>>>>>>>>>>>")
        training = TrainingPipeline()
        training.main()
        logger.info(f">>>>>>>>>>> The {STAGE_NAME} has completed successfully >>>>>>>>>>")
    except Exception as e:
        raise e