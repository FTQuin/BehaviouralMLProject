import os
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
        return tf.convert_to_tensor(np.array(frames))


class trainingAbstract:
    def __init__(self, dataset_name, extractor, train_test_split):
        self.dataset_name = dataset_name
        self.extractor = extractor
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
        def __init__(self, extractor, train_test_split=.75, file_name='features'):
            super(UCF.training, self).__init__(UCF.dataset_name, extractor, train_test_split)

            # get path
            dir_path = os.path.join(self.features_save_path,  # path to features
                                    file_name + '.zip')  # specific sub set of features
            # load into pandas df
            self.data = pd.read_csv(dir_path, compression='zip')

            train_vids = pd.Series(self.data['video'].unique()).sample(frac=train_test_split)
            self.train_data = self.data.loc[self.data['video'].isin(train_vids)]
            self.train_labels = self.train_data.groupby('video')['label'].first().values
            self.test_data = self.data.loc[~self.data['video'].isin(train_vids)]
            self.test_labels = self.test_data.groupby('video')['label'].first().values

        def get_train_data(self, seq_len):
            return self._get_data(self.train_data, seq_len)

        def get_train_labels(self):
            return self._get_labels(self.train_labels)

        def get_test_data(self, seq_len):
            return self._get_data(self.test_data, seq_len)

        def get_test_labels(self):
            return self._get_labels(self.test_labels)

        def _get_data(self, df, seq_len):
            def prepare_row(f):
                # drop non-feature columns
                g = f.drop(['video', 'label', 'frame'], axis=1)
                arr = g.to_numpy()

                # get random starting point
                start = np.random.randint((len(arr) - seq_len + 1) if len(arr) > seq_len else 1)
                arr = arr[start:start + seq_len]

                # pad with 0s
                out = np.zeros((seq_len, self.extractor.num_features))
                out[seq_len - len(arr):] = arr
                return out

            feat_arr = np.array(list(df.groupby('video').apply(prepare_row)))
            return feat_arr

        def _get_labels(self, df):
            classes = np.unique(df)
            pos = list(map(lambda x: np.where(classes == x)[0][0], df))
            out = np.array(pos).reshape(-1, 1)
            return out


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
