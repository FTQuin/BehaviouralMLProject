import datetime
import os

import cv2
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import frame_feature_extractor as ffe
import prepare_data as pd

NUM_FEATURES = 17*6*3
SEQ_LENGTH = 20

EPOCHS = 10

MODEL_FILE = 'models/jump_skate_20frame_10epochs'
FEATURE_FILE = 'processed_features/processed_data_20.npz'


# load data from feature file
def load_features(feature_path=FEATURE_FILE):
    features, labels, label_names = [item[1] for item in np.load(feature_path).items()]
    print("\nLoaded Features")
    return features, labels, label_names


# Utility for our sequence model.
def create_sequence_model(num_labels=2):
    # if label_processor is not None:
    #     class_vocab = label_processor.get_vocabulary()
    #     num_labels = len(class_vocab)

    frame_features_input = keras.Input((SEQ_LENGTH, NUM_FEATURES))

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    output = keras.layers.Dense(num_labels, activation='softmax')(x)

    rnn_model = keras.Model([frame_features_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# Utility for running experiments.
def train_model():
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_FILE, save_weights_only=True, save_best_only=False, verbose=1
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    seq_model = create_sequence_model()
    seq_model.fit(
        features,
        labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint, tensorboard_callback],
    )
    print('Training Done')


# Get a trained model from a file
def get_trained_sequence_model(filepath=MODEL_FILE, num_labels=2):
    seq_model = create_sequence_model(num_labels)
    seq_model.load_weights(filepath)
    return seq_model


if __name__ == '__main__':
    # load features
    features, labels, label_names = load_features()

    # create label processor
    label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=label_names)
    print('Labels:', label_processor.get_vocabulary())

    # train model
    train_model()
