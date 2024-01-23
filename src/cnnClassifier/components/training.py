import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    

    
    
    # Perform Data Processing on the train, val, test dataset
    def train(self):
        data_preprocess = tf.keras.Sequential(
        name="data_preprocess",
        layers=[
            tf.keras.layers.Resizing(self.config.height, self.config.width), # Shape Preprocessing
            tf.keras.layers.Rescaling(1.0/255), # Value Preprocessing
        ]
        )
        train_ds = self.config.train_ds.map(lambda x, y: (data_preprocess(x), y))
        val_ds = self.config.val_ds.map(lambda x, y: (data_preprocess(x), y))
        test_ds = self.config.test_ds.map(lambda x, y: (data_preprocess(x), y))
        

        pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=[self.config.height,self.config.width, 3])
        pretrained_model.trainable=False
        vgg16_model = tf.keras.Sequential([
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.config.num_classes, activation='softmax')
        ])
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("final_model.h5", save_best_only=True)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",patience=5, restore_best_weights=True
        )

        vgg16_model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        history_vgg16 = vgg16_model.fit(train_ds, epochs=self.config.EPOCHS, validation_data=val_ds,callbacks=[checkpoint_callback,early_stopping_callback])
        vgg16_model.save(self.config.trained_model_path)