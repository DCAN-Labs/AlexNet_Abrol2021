from tensorflow import keras


def get_alex_net():
    return keras.models.Sequential([
        keras.layers.Conv3D(filters=96, kernel_size=(11, 11, 11), strides=(4, 4, 4), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),
        keras.layers.Conv3D(filters=256, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),
        keras.layers.Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=1)
    ])
