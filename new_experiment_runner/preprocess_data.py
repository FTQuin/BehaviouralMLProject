import os

import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

import feature_extractors as pre

VIDEO_DIRECTORY = '../datasets/UCF-101'
FEATURE_FILE = "../TFlite-NN/processed_features/processed_data_20.npz"

SEQ_LENGTH = 20
NUM_FEATURES = 56*6

feature_extractor = pre.movenet_extractor()

# Load Videos
def load_video(path, max_frames=0):
    # TODO: make docstring
    """
    description ...

    :parm p1: parameter description
    :type p1: parameter type
    :return: return description
    :rtype: return type
    """
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

# get video paths and use directories as label names
def prepare_all_videos():
    video_paths = [(os.path.split(a)[1], files) for a, b, files in os.walk(VIDEO_DIRECTORY)]
    video_paths = [[(l[0], i) for i in l[1]] for l in video_paths[1:]]

    # put into a dataFrame
    video_label_df = pd.DataFrame()
    for i in video_paths:
        video_label_df = video_label_df.append(i, ignore_index=True)
    video_label_df = video_label_df.rename({0: 'label', 1: 'video_path'}, axis=1)

    # get label processor
    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(video_label_df['label'])
    )
    label_names = label_processor.get_vocabulary()
    print('Labels:', label_names)

    num_samples = len(video_label_df)
    video_paths = video_label_df['video_path'].values.tolist()
    labels = video_label_df['label'].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video, get features
    for idx, path in enumerate(video_label_df.values[0:2]):
        path = os.path.join(path[0], path[1])
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(VIDEO_DIRECTORY, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # display progress
        print(f'Extracting features of video: {idx}/{num_samples}, {100*idx/num_samples:.2f}% done')

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(SEQ_LENGTH, video_length)
            for j in range(length):
                features = feature_extractor.extract(tf.expand_dims(tf.constant(batch[j], dtype='uint8'), axis=0))
                temp_frame_features[i, j, :] = keras.layers.Flatten()(features)
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx, ] = temp_frame_features.squeeze()
        frame_masks[idx, ] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels, label_names


if __name__ == '__main__':
    print('Preparing Data')
    # prepare data
    features_masks, labels, label_names = prepare_all_videos()
    features = features_masks[0]

    # save data
    np.savez(FEATURE_FILE, features=features, labels=labels, label_names=label_names)
    print('Features Saved')