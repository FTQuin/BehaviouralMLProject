"""
:Date: 2022-02-28
:Author: Quin Adam, Govind Tanda
:Description: Feature extractor class for 3 extractors (Movenet, MobilenetV2, InceptionV3)
"""

import numpy as np
import tensorflow as tf
from abc import abstractmethod


class ExtractorAbstract:
    """
    Abstract class for feature extractors

    Attributes:
        :param num_features: Number of features the extractor will extract
        :type num_features: int
        :param name: Name for the extractor
        :type name: String

    Methods:
        pre_process_extract_video(frames)
        train_extract(features)
        live_extract(frame)
    """

    def __init__(self, num_features, name):
        self.num_features = num_features
        self.name = name

    @abstractmethod
    def pre_process_extract_video(self, frames):
        pass

    @abstractmethod
    def live_extract(self, frame):
        pass


class MovenetExtractor(ExtractorAbstract):
    """
    MovenetExtractor class

    Attributes:
        :param threshold: Prediction confidence scores of each of the 17 keypoints movenet extracts
        :type threshold: float (0.0 - 1.0)
    """

    num_features = 6 * 56
    name = "MovenetExtractor"

    def __init__(self, threshold=0.0):
        """ MovenetExtractor Constructor """
        super(MovenetExtractor, self).__init__(MovenetExtractor.num_features, MovenetExtractor.name)

        # get movenet
        self.model = tf.saved_model.load('movenet/movenet_multipose_lightning_1')  # keep ref to model
        self.movenet = self.model.signatures['serving_default']

        # threshold when outputting features
        self.threshold = threshold

    def extract_frame(self, frame):
        """
        :param frame: Video frame
        :return: 17 keypoint data scores from the frame
        """
        t1 = tf.image.resize_with_pad(frame, 256, 256)  # resize and pad
        t2 = tf.cast(t1, dtype=tf.int32)  # cast to int32
        return self.movenet(t2)['output_0']  # get result

    def pre_process_extract_video(self, frames):
        """
        :param frames: Frames array for current video
        :return: Preprocessed frames with 0 threshold value to extract all 17 keypoints
        """
        return tf.map_fn(self.extract_frame, tf.expand_dims(frames, axis=1),
                         fn_output_signature=tf.TensorSpec((1, 6, 56)),
                         )

    def live_extract(self, frame):
        """
        :param frame: Single video frame
        :return: Frame with extracted features
        """
        t1 = self.extract_frame(tf.expand_dims(frame, axis=0))
        cond = tf.less(t1, tf.constant(self.threshold, dtype=t1.dtype))
        out = tf.where(cond, tf.zeros(tf.shape(t1), dtype=t1.dtype), t1)
        return tf.reshape(out, (1, -1))  # flatten


class InceptionExtractor(ExtractorAbstract):
    """ InceptionExtractor class """

    num_features = 2048
    name = "InceptionExtractor"

    def __init__(self):
        """ InceptionExtractor Constructor """
        super(InceptionExtractor, self).__init__(InceptionExtractor.num_features, InceptionExtractor.name)
        self.feature_extractor = tf.keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )

    def pre_process_extract_video(self, frames, batch_size=32):
        """
        :param frames: Array of video frames
        :param batch_size: Number of samples to work on in current iteration
        :return: Preprocessed frames using InceptionV3.preprocess_input (scale values between -1, 1)
        """
        batches = np.split(frames, [i for i in range(batch_size, len(frames), batch_size)])
        out = None
        for batch in batches:
            pre = tf.keras.applications.InceptionV3.preprocess_input(tf.cast(batch, dtype='float32'))
            res = self.feature_extractor(pre)
            if out is not None:
                out = tf.concat([res, out], axis=0)
            else:
                out = res
        return out

    def live_extract(self, frame):
        """
        :param frame: Single video frame
        :return: Extracted features
        """
        t1 = tf.expand_dims(frame, axis=0)
        t2 = tf.keras.applications.InceptionV3.preprocess_input(tf.cast(t1, dtype='float32'))
        out = self.feature_extractor(t2)
        return out


class MobileNetV2Extractor(ExtractorAbstract):
    """ MobileNetV2Extractor class """

    num_features = 1280
    name = "MobileNetV2Extractor"

    def __init__(self):
        """ MobileNetV2 Extractor Constructor """
        super(MobileNetV2Extractor, self).__init__(MobileNetV2Extractor.num_features, MobileNetV2Extractor.name)
        self.feature_extractor = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )

    def pre_process_extract_video(self, frames, batch_size=32):
        """
        :param frames: Array of video frames
        :param batch_size: Number of samples to work on in current iteration
        :return: Preprocessed frames using mobilenet.preprocess_input (scale values between -1, 1)
        """
        batches = np.split(frames, [i for i in range(batch_size, len(frames), batch_size)])
        out = None
        for batch in batches:
            pre = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(batch, dtype='float32'))
            res = self.feature_extractor(pre)
            if out is not None:
                out = tf.concat([res, out], axis=0)
            else:
                out = res
        return out

    def live_extract(self, frame):
        """
        :param frame: Single video frame
        :return: Extracted features
        """
        t1 = tf.expand_dims(frame, axis=0)
        t2 = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(t1, dtype='float32'))
        out = self.feature_extractor(t2)
        return out


if __name__ == '__main__':
    img = tf.io.decode_jpeg(tf.io.read_file('../TFlite-NN/images/image.jpeg'))
    img = tf.expand_dims(img, axis=0)
    movenet = MovenetExtractor()
    res = movenet.pre_process_extract_video(img)
    print(res.shape)
