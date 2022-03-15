"""
:Date: 2022-02-28
:Author: Quin Adam
:Description: Extracts features from a frame
"""
import os

import cv2
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import frame_feature_extractor as ffe

feature_extractor = fec.mobile
dataset = datasets.UCF


#[video, x, y, channle]
#[video, label], [video, frames, features]]


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


# extract features from videos
def prepare_all_videos():
    # TODO: make docstring
    """
    description ...

    :parm p1: parameter description
    :type p1: parameter type
    :return: return description
    :rtype: return type
    """

    frame_features = np.zeros(
        shape=(num_samples, SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video, get features
    for idx, path in enumerate(video_label_df.values):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(VIDEO_DIRECTORY, path))
        frames = frames[None, ...]
        # display progress
        print(f'Extracting features of video: {idx}/{num_samples}, {100*idx/num_samples:.2f}% done')

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            for j in range(video_length):
                features = ffe.get_features_from_image(batch[j, :])
                temp_frame_features[i, j, :] = tf.keras.layers.Flatten()(features)
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx, ] = temp_frame_features.squeeze()

    return frame_features, labels


if __name__ == '__main__':
    print('Preparing Data')
    # prepare data
    features, labels = prepare_all_videos()

    # save data
    save_data(features, labels)
