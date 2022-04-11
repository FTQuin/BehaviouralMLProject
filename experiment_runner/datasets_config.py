"""
:Date: 2022-02-28
:Author: Quin Adam, Govind Tanda
:Description: Module for extracting features from training examples and serving those features for training
"""
import os

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd


class Dataset:
    class Preprocessing:
        def __init__(self, dataset_path, extractor):
            """

            :param dataset_path:
            :type dataset_path:
            :param extractor:
            :type extractor:
            """
            self.extractor = extractor

            self._raw_data_path = os.path.abspath(dataset_path)
            self.features_save_path = os.path.abspath(os.path.join('../features',
                                                                   os.path.split(dataset_path)[-1],
                                                                   extractor.name))

            # get video paths
            self._video_labels_paths = [(curr_path, files) for curr_path, sub_dirs, files in os.walk(self._raw_data_path)]
            self._video_labels_paths = [(os.path.split(label_tup[0])[1], os.path.join(label_tup[0], vid_path))
                                        for label_tup in self._video_labels_paths[1:] for vid_path in label_tup[1]]

            # iterate through raw data
            class VideoIterator:
                def __init__(self_iter, video_labels_paths):
                    self_iter.video_labels_paths = video_labels_paths

                def __iter__(self_iter):
                    self_iter.video_index = 0
                    return self_iter

                def __next__(self_iter):
                    # if self_iter.video_index < 20:  # uncomment to limit number of vids
                    if self_iter.video_index < len(self_iter.video_labels_paths):
                        x = self_iter.video_labels_paths[self_iter.video_index]
                        self_iter.video_index += 1
                        return {'label': x[0], 'frames': self._load_video(x[1]), 'name': os.path.split(x[1])[-1]}
                    else:
                        raise StopIteration

            # return iterator
            self.data_iterator = iter(VideoIterator(self._video_labels_paths))

        @staticmethod
        def _load_video(path):
            frames = []
            cap = cv2.VideoCapture(path)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = frame[:, :, [2, 1, 0]]
                    frame = frame.astype('int32')
                    frames.append(frame)
            finally:
                cap.release()
            return np.array(frames)

    class Training:
        def __init__(self, dataset_path, extractor, seq_len, train_test_split=.75, enable_caching=False):
            tf.random.set_seed(1)
            self.extractor = extractor
            self.seq_len = seq_len
            self.train_test_split = train_test_split

            self._raw_data_path = os.path.abspath(dataset_path)
            self._features_save_path = os.path.abspath(os.path.join('../features',
                                                                    os.path.split(dataset_path)[-1],
                                                                    extractor.name))
            self.labels = list(os.walk(self._features_save_path))[0][1]
            if '.cache' in self.labels:
                self.labels.remove('.cache')

            # ==== CREATE DATASET ====
            dataset = tf.data.Dataset.list_files(os.path.join(self._features_save_path, '*/*.zip'))
            dataset = dataset.shuffle(10000000, seed=1)

            # train test split by file
            split = round(1/(1-self.train_test_split))  # put every nth example in validation split
            train_dataset = dataset.window(split, split + 1).flat_map(lambda ds: tf.data.Dataset.zip((ds)))
            validation_dataset = dataset.skip(split).window(1, split + 1).flat_map(lambda ds: tf.data.Dataset.zip((ds)))

            def process_path(file_path):
                def sub(fp):
                    df = pd.read_csv(fp.numpy().decode('utf-8'))
                    label = self.labels.index(df['label'].values[0])
                    df = df.drop(['video', 'label', 'frame', 'Unnamed: 0'], axis=1)
                    arr = df.to_numpy()
                    return tf.convert_to_tensor(arr, dtype='float32'), tf.convert_to_tensor([label]*len(arr), dtype='float32')

                res, labels = tf.py_function(
                    sub,
                    [file_path],
                    [tf.TensorSpec(shape=[None, self.extractor.num_features], dtype=tf.dtypes.float32),
                     tf.TensorSpec(shape=[self.extractor.num_features], dtype=tf.dtypes.float32)])
                res.set_shape([None, self.extractor.num_features])
                labels.set_shape([None])

                ds = tf.data.Dataset.from_tensor_slices(res)
                ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(labels)))

                ds = ds.window(self.seq_len, shift=1, drop_remainder=True)
                ds = ds.shuffle(buffer_size=1000000, seed=1)
                ds = ds.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(self.seq_len), y.batch(1))))

                return ds

            # window datasets
            train_dataset = train_dataset.interleave(process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
            validation_dataset = validation_dataset.interleave(process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

            # shuffle
            train_dataset = train_dataset.shuffle(buffer_size=1000, seed=1)
            validation_dataset = validation_dataset.shuffle(buffer_size=1000, seed=1)

            # batch
            self.train_dataset = train_dataset.batch(64, num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)
            self.validation_dataset = validation_dataset.batch(64, num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)

            if enable_caching:
                cache_path = os.path.join(self._features_save_path, '.cache/')
                try: os.mkdir(cache_path)
                except FileExistsError: pass

                self.train_dataset = self.train_dataset.cache(cache_path)
                self.validation_dataset = self.validation_dataset.cache(cache_path)
