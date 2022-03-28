from tensorflow import keras


class GRU:
    @staticmethod
    def gru1(output_size, activation_function='relu',
             loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        model = keras.Sequential(
            keras.layers.GRU(16, return_sequences=True),
            keras.layers.GRU(8),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(16, activation=activation_function),
            keras.layers.Dense(8, activation=activation_function),
            keras.layers.Dense(output_size, activation='softmax')
        )

        model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
        return model

    @staticmethod
    def gru2(output_size, activation_function='relu',
             loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        model = keras.Sequential(
            [
                keras.layers.GRU(160, return_sequences=True),
                keras.layers.GRU(80),
                keras.layers.Dropout(0.4),
                keras.layers.Dense(160, activation=activation_function),
                keras.layers.Dense(80, activation=activation_function),
                keras.layers.Dense(16, activation=activation_function),
                keras.layers.Dense(output_size, activation='softmax'),
            ]
        )

        model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
        return model


class LSTM:
    @staticmethod
    def lstm1(output_size, activation_function='relu',
              loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        model = keras.Sequential(
            keras.layers.LSTM(16, return_sequences=True),
            keras.layers.LSTM(8),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(16, activation=activation_function),
            keras.layers.Dense(8, activation=activation_function),
            keras.layers.Dense(output_size, activation='softmax')
        )

        model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
        return model

    @staticmethod
    def lstm2(output_size, activation_function='relu',
              loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        model = keras.Sequential(
            [
                keras.layers.LSTM(256, return_sequences=True),
                keras.layers.LSTM(128),
                keras.layers.Dropout(0.4),
                keras.layers.Dense(256, activation=activation_function),
                keras.layers.Dense(128, activation=activation_function),
                keras.layers.Dense(64, activation=activation_function),
                keras.layers.Dense(output_size, activation='softmax'),
            ]
        )

        model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
        return model
