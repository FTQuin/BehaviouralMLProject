import datetime
import os

import cv2
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import frame_feature_extractor as ffe

IMG_SIZE = 256

MAX_SEQ_LENGTH = 250
NUM_FEATURES = 17*6*3
EPOCHS = 100


## Load Videos
def load_video(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, [2, 1, 0]]
            frame = frame.astype('float64')
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


x = [(os.path.split(a)[1], files) for a, b, files in os.walk('../TFlite-NN/UCF-101')]
y = [[(l[0], i) for i in l[1]] for l in x[1:]]

all_vids_df = pd.DataFrame()
for i in y:
    all_vids_df = all_vids_df.append(i, ignore_index=True)
all_vids_df = all_vids_df.rename({0: 'tag', 1: 'video_name'}, axis=1)

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(all_vids_df["tag"])
)
print(label_processor.get_vocabulary())


def prepare_all_videos(root_dir):
    num_samples = sum(len(files) for a, b, files in os.walk(root_dir))
    video_paths = all_vids_df["video_name"].values.tolist()
    labels = all_vids_df["tag"].values
    labels = label_processor(labels[..., None]).numpy()
    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(all_vids_df.values):
        path = os.path.join(path[0], path[1])
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # if idx > 1:
        #     break
        print("Extracting features of video:", idx, "/", num_samples)

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = ffe.get_features_from_image(
                    batch[j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


# # # prepare data
# # features_masks, labels = prepare_all_videos("UCF-101")
# # features = features_masks[0]
#
# # save data
# np.savez("processed_data", features=features, labels=labels)
# print("dumped")

# load data
features, labels= [item[1] for item in np.load("processed_data.npz").items()]
print("loaded")


# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    output = keras.layers.Dense(len(class_vocab), activation='softmax')(x)

    rnn_model = keras.Model([frame_features_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# Utility for running experiments.
def run_experiment():
    filepath = "tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=False, verbose=1
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    seq_model = get_sequence_model()
    history = seq_model.fit(
        features,
        labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint, tensorboard_callback],
    )

    seq_model.load_weights(filepath)
    # _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()

print('\n\n******** END ********')
