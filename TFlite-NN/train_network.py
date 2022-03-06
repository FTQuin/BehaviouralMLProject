"""
:Date: 2022-02-28
:Author: Quin Adam
:Description: Extracts features from a frame
"""
import datetime

import numpy as np
from tensorflow import keras
import tensorflow as tf

NUM_FEATURES = 17*6*3
SEQ_LENGTH = 20

EPOCHS = 25

MODEL_FILE = 'models/9Sports_100Frame_25epochs'
FEATURE_FILE = 'processed_features/processed_data_9Sports_100Frame.npz'


# load data from feature file
def load_features(feature_path=FEATURE_FILE):
    # TODO: make docstring
    """
    description ...

    :parm p1: parameter description
    :type p1: parameter type
    :return: return description
    :rtype: return type
    """
    features, labels, label_names = [item[1] for item in np.load(feature_path).items()]
    print("\nLoaded Features")
    return features, labels, label_names


# Utility for our sequence model.
def create_sequence_model(num_labels=2):
    # TODO: make docstring
    """
    description ...

    :parm p1: parameter description
    :type p1: parameter type
    :return: return description
    :rtype: return type
    """
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
    # TODO: make docstring
    """
    description ...

    :parm p1: parameter description
    :type p1: parameter type
    :return: return description
    :rtype: return type
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_FILE, save_weights_only=True, save_best_only=False, verbose=1
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    seq_model = create_sequence_model(len(np.unique(label_names)))
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
    # TODO: make docstring
    """
    description ...

    :parm p1: parameter description
    :type p1: parameter type
    :return: return description
    :rtype: return type
    """
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
