"""
:Date: 2022-02-28
:Author: Quin Adam
:Description: Extracts features from a frame
"""
import os

import pandas as pd
import tensorflow as tf

import feature_extractors_config as fec
import datasets_config

# ==== MODIFIABLE PARAMS ====
FEATURE_EXTRACTOR = fec.MobileNetV2Extractor()
DATASET_PATH = '../datasets/UCF-3'

# init dataset
dataset = datasets_config.Dataset.Preprocessing(DATASET_PATH, FEATURE_EXTRACTOR)


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

    # For each video, get features
    for idx, video_info in enumerate(dataset.data_iterator):
        video_frame_features = prepare_one_video(video_info)

        print(f'Extracted features from video {idx}')

        # make dir for video
        try: os.makedirs(os.path.join(dataset.features_save_path, video_frame_features['label'][0]))
        except FileExistsError: pass

        # save features
        video_frame_features.to_csv(os.path.join(dataset.features_save_path, video_frame_features['label'][0],
                                                 video_frame_features['video'][0].replace('.avi', '.zip')),
                                    compression=dict(method='zip',
                                                     archive_name=video_frame_features['video'][0].replace('.avi',
                                                                                                           '.csv')))

        print(f'Saved to  {os.path.join(dataset.features_save_path, video_frame_features["label"][0], video_frame_features["video"][0].replace(".avi", ".csv"))}')


def prepare_one_video(video_info):
    frames = video_info['frames']

    res = FEATURE_EXTRACTOR.pre_process_extract_video(frames)
    res = tf.reshape(res, (res.shape[0], -1))
    temp_frame_features = pd.DataFrame(data=res.numpy())

    temp_frame_features = pd.DataFrame(
        data={'video': video_info['name'], 'label': video_info['label'], 'frame': range(len(frames)),
              **temp_frame_features})

    return temp_frame_features


if __name__ == '__main__':
    print('Preparing Data')
    # prepare data
    features = prepare_all_videos()

    # TODO: make more verbose
    print('DONE')
