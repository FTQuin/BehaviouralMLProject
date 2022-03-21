import os
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd


class DatasetAbstract:
    dataset_name = 'ABSTRACT'
    _raw_data_path = None
    _features_save_path = None
    _data_iterator = None

    def __init__(self, train_test_split, extractor):
        self.train_test_split = train_test_split
        self.extractor = extractor
        self.labels = []

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
        return tf.convert_to_tensor(np.array(frames))

    def raw_data_path(self):
        if not self._raw_data_path:
            self._raw_data_path = os.path.abspath(os.path.join('../datasets', self.dataset_name))
        return self._raw_data_path

    def features_save_path(self):
        if not self._features_save_path:
            self._features_save_path = os.path.abspath(os.path.join('../datasets', self.dataset_name))
        return self._features_save_path

    def data_iterator(self):
        # returns iterator that returns label, video[frame, x, y, c]
        pass


class UCF(DatasetAbstract):
    dataset_name = 'UCF-101'
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

    def train_data(self, seq_len):
        # dir_path = os.path.join(self.features_save_path(), str(type(self.extractor)).split('.')[-1][:-2])
        dir_path = 'D:\\Users\\quinj\\Documents\\School\\Winter 2022\\Capstone COMP 4910\\BehaviouralMLProject\\features\\UCF-101\\MovenetExtractor\\features.zip'
        data = pd.read_csv(dir_path, compression='zip')
        keys = data['video'].unique()
        out = np.ndarray((len(keys), seq_len, int(data.columns[-1]) + 1))
        for idx, key in enumerate(keys):
            df = data[data['video'] == key]
            self.labels.append(df['label'].unique()[0])
            arr = df.drop(['video', 'label', 'frame'], axis=1).to_numpy()
            start = np.random.randint(len(arr) - seq_len + 1)
            arr = arr[start:start + seq_len]
            out[idx] = arr

        return out

    def train_labels(self):
        # dir_path = os.path.join(self.features_save_path(), str(type(self.extractor)).split('.')[-1][:-2])
        # dir_path = 'D:\\Users\\quinj\\Documents\\School\\Winter 2022\\Capstone COMP 4910\\BehaviouralMLProject\\features\\UCF-101\\MovenetExtractor\\features.zip'
        # data = pd.read_csv(dir_path, compression='zip')
        # labels = data['label']
        # classes = data['label'].unique()
        # out = np.ndarray((len(labels), len(classes)))
        # return labels
        classes = np.unique(self.labels)
        pos = list(map(lambda x: np.where(classes == x)[0][0], self.labels))
        return np.array(pos).reshape(-1, 1)
        # return tf.one_hot(pos, len(classes))


class NTU(DatasetAbstract):
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
