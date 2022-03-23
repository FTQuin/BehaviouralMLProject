from tensorflow import keras


class GRU:
    @staticmethod
    def gru1(input_shape, output_size, seq_len=-1, activation_function='relu',
             loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        frame_features_input = keras.Input((seq_len, input_shape))
        x = keras.layers.GRU(16, return_sequences=True)(
            frame_features_input
        )
        x = keras.layers.GRU(8)(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(16, activation=activation_function)(x)
        x = keras.layers.Dense(8, activation=activation_function)(x)
        output = keras.layers.Dense(output_size, activation='softmax')(x)

        gru_model = keras.Model([frame_features_input], output)

        gru_model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

        return gru_model

    @staticmethod
    def gru2(input_shape, output_size, seq_len=-1, activation_function='relu',
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
    def lstm1(input_shape, output_size, seq_len=-1, activation_function='relu',
              loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        frame_features_input = keras.Input((seq_len, input_shape))
        x = keras.layers.LSTM(16, return_sequences=True)(frame_features_input)
        x = keras.layers.LSTM(8)(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(16, activation=activation_function)(x)
        x = keras.layers.Dense(8, activation=activation_function)(x)
        output = keras.layers.Dense(output_size, activation='softmax')(x)

        lstm_model = keras.Model([frame_features_input], output)

        lstm_model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

        return lstm_model

    @staticmethod
    def lstm2(input_shape, output_size, seq_len=-1, activation_function='relu',
              loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        frame_features_input = (input_shape, seq_len)
        model = keras.Sequential(
            [
                keras.layers.LSTM(256, input_shape=frame_features_input, return_sequences=True),
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
