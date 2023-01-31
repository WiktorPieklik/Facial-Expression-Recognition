import tensorflow as tf
from typing import Tuple, List
from src.config import IMG_DIR, IMG_SHAPE


def get_train_val_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds_path = str((IMG_DIR / "Training").resolve())
    validation_ds_path = str((IMG_DIR / "PublicTest").resolve())

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_ds_path,
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=None,
        image_size=IMG_SHAPE,
        seed=2137,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_ds_path,
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=None,
        image_size=IMG_SHAPE,
        seed=2137,
    )

    return train_ds, val_ds


class CustomNet:
    def __init__(self, epochs: int, batch_size: int, logs_path: str):
        self._epochs = epochs
        self._batch_size = batch_size
        self._model: tf.keras.Model = self.__init_model()
        self._callbacks: List[tf.keras.callbacks] = [
            tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath=f"{logs_path}/best.h5"),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
            tf.keras.callbacks.CSVLogger(filename=f"{logs_path}/history.csv"),
        ]

    def train(self):
        rescale = tf.keras.layers.Rescaling(1. / 255)
        train_ds, val_ds = get_train_val_datasets()
        train_ds = train_ds \
            .batch(self._batch_size)\
            .map(lambda img, label: (rescale(img), label))

        val_ds = train_ds \
            .batch(self._batch_size)\
            .map(lambda img, label: (rescale(img), label))
        gpus_count = tf.config.list_physical_devices("GPU")
        if len(gpus_count) > 0:
            print(f"{self.__class__.__name__}: using GPU")
            with tf.device("/GPU:0"):
                history = self.__fit(train_ds, val_ds)
        else:
            print(f"{self.__class__.__name__}: using CPU")
            history = self.__fit(train_ds, val_ds)

        return history

    def __fit(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset):
        self.__compile_model()
        return self._model.fit(
                    train_ds,
                    epochs=self._epochs,
                    callbacks=self._callbacks,
                    validation_data=val_ds
                )

    @staticmethod
    def __init_model() -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=IMG_SHAPE + (1,)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((1, 1)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(512, kernel_size=(5, 5), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(512, kernel_size=(5, 5), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(512, kernel_size=(5, 5), activation="relu", padding="same",
                                   kernel_initializer="glorot_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(7, activation="softmax")
        ])
        model.summary()

        return model

    def __compile_model(self) -> None:
        self._model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
        )
