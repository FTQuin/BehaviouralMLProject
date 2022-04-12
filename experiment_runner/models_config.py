"""
:Date: 2022-02-28
:Author: Quin Adam, Govind Tanda
:Description: Model classes
"""

from tensorflow import keras


class GRU:
    """
    Class for GRU Models, create your gru model as a function or modify existing models

    All methods return sequential GRU models
    """
    @staticmethod
    def gru1(output_size, activation_function='relu',
             loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        model = keras.Sequential([
            keras.layers.GRU(64),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256, activation=activation_function),
            keras.layers.Dense(128, activation=activation_function),
            keras.layers.Dense(output_size, activation='softmax'),
        ])

        model.compile(loss=loss_function, optimizer=optimizer, metrics=["sparse_categorical_accuracy"])
        return model

    @staticmethod
    def gru2(output_size, activation_function='relu',
              loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        model = keras.Sequential([
            keras.layers.GRU(1024, return_sequences=True),
            keras.layers.Dropout(0.4),
            keras.layers.GRU(512),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1024, activation=activation_function),
            keras.layers.Dense(512, activation=activation_function),
            keras.layers.Dense(128, activation=activation_function),
            keras.layers.Dense(output_size, activation='softmax'),
        ])

        model.compile(loss=loss_function, optimizer=optimizer, metrics=["sparse_categorical_accuracy"])
        return model


class LSTM:
    """
    Class for LSTM Models, create your LSTM model as a function or modify existing models

    All methods return sequential LSTM models
    """
    @staticmethod
    def lstm1(output_size, activation_function='relu',
             loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        model = keras.Sequential([
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256, activation=activation_function),
            keras.layers.Dense(128, activation=activation_function),
            keras.layers.Dense(output_size, activation='softmax'),
        ])

        model.compile(loss=loss_function, optimizer=optimizer, metrics=["sparse_categorical_accuracy"])
        return model

    @staticmethod
    def lstm2(output_size, activation_function='relu',
              loss_function="sparse_categorical_crossentropy", optimizer="adam"):
        model = keras.Sequential([
            keras.layers.LSTM(1024, return_sequences=True),
            keras.layers.Dropout(0.4),
            keras.layers.LSTM(512),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1024, activation=activation_function),
            keras.layers.Dense(512, activation=activation_function),
            keras.layers.Dense(128, activation=activation_function),
            keras.layers.Dense(output_size, activation='softmax'),
        ])

        model.compile(loss=loss_function, optimizer=optimizer, metrics=["sparse_categorical_accuracy"])
        return model
