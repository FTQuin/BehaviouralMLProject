import os

import pandas
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd


class preprocessingAbstract:
    def __init__(self, dataset_name, extractor):
        self.dataset_name = dataset_name
        self.extractor = extractor
        self.features_save_path = os.path.abspath(os.path.join('../features', self.dataset_name, extractor.name))

        self._raw_data_path = os.path.abspath(os.path.join('../datasets', self.dataset_name))

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


class trainingAbstract:
    def __init__(self, dataset_name, extractor, seq_len, train_test_split):
        self.dataset_name = dataset_name
        self.extractor = extractor
        self.seq_len = seq_len
        self.features_save_path = os.path.abspath(os.path.join('../features', self.dataset_name, self.extractor.name))

        self.train_test_split = train_test_split


class UCF:
    dataset_name = 'UCF-101'

    class preprocessing(preprocessingAbstract):
        def __init__(self, extractor):
            super(UCF.preprocessing, self).__init__(UCF.dataset_name, extractor)

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
            self.data_iterator = iter(VideoIterator(self._video_labels_paths))

    class training(trainingAbstract):
        def __init__(self, extractor, seq_len, train_test_split=.75):
            super(UCF.training, self).__init__(UCF.dataset_name, extractor, seq_len, train_test_split)

            self.labels = list(os.walk(self.features_save_path))[0][1]

            dataset = tf.data.Dataset.list_files(os.path.join(self.features_save_path, '*/*.zip'))
            dataset = dataset.shuffle(10000000)

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
                # ds = ds.shuffle(buffer_size=1000)
                ds = ds.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(self.seq_len), y.batch(1))))

                return ds

            dataset = dataset.interleave(process_path)

            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(64)

            self.dataset = dataset


class NTU:
    dataset_name = 'NTU'
    __video_labels_paths = None

    def data_iterator(self):
        if not self._data_iterator:
            class VideoIterator:
                def __init__(self_iter, vlp):
                    self_iter.video_labels_paths = vlp

                def __iter__(self_iter):
                    self_iter.video_index = 0
                    return self_iter

                def __next__(self_iter):
                    # TODO: uncomment
                    # if self_iter.video_index < len(video_labels_paths):
                    if self_iter.video_index < 20:
                        x = self_iter.video_labels_paths[self_iter.video_index]
                        self_iter.video_index += 1
                        return {'label': x[0], 'frames': self._load_video(x[1]), 'name': os.path.split(x[1])[-1]}
                    else:
                        raise StopIteration

            self._data_iterator = iter(VideoIterator(self._video_labels_paths()))
        # end_if
        return self._data_iterator

    def _video_labels_paths(self):
        if not self.__video_labels_paths:
            # get video paths
            self.__video_labels_paths = [(curr_path, files) for curr_path, sub_dirs, files in os.walk(self.raw_data_path())]
            self.__video_labels_paths = [(os.path.split(label_tup[0])[1], os.path.join(label_tup[0], vid_path)) for label_tup
                                  in self.__video_labels_paths[1:] for vid_path in label_tup[1]]

        return self.__video_labels_paths
