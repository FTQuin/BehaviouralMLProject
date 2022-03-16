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

# import frame_feature_extractor as ffe
from feature_extractors_config import movenet_extractor

# dataset = datasets.UCF

# [label], [video,frame, x, y, channel]
# [video, label], [video, frames, features]]

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

    # pass frame_features into model
    frame_features = pd.DataFrame()

    # # For each video, get features
    for idx, video, label in enumerate(dataset.get_video_information()):
        # Gather all its frames and add a batch dimension.
        frames = video['frames'][None, ...]

        temp_frame_features = pd.DataFrame()
        # display progress
        print(f'Extracting features of video: {idx}/{dataset.num_videos}, {100 * idx / dataset.num_videos:.2f}% done')

        # Extract features from the frames of the current video.
        video_length = frames.shape[0]
        for j in range(video_length):
            extracted_features = movenet_extractor.extract(frames[j])
            series = pd.Series(data=tf.keras.layers.Flatten()(extracted_features))
            temp_frame_features = pd.concat((temp_frame_features, series))

        temp_frame_features = pd.DataFrame(
            data={'video': video['name'], 'label': label, 'frame': range(video_length), **temp_frame_features})
        frame_features = pd.concat((frame_features, temp_frame_features))

    return frame_features


def save_data(extracted_frame_pd):
    extracted_frame_pd.to_csv(datasets.save_file_path, index=False)

if __name__ == '__main__':
    print('Preparing Data')
    # prepare data
    features = prepare_all_videos()

    # save data
    save_data(features)
