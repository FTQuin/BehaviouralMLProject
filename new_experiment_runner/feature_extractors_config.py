import numpy as np
import tensorflow as tf
from abc import abstractmethod


class ExtractorAbstract:
    def __init__(self, num_features, name):
        self.num_features = num_features
        self.name = name

    @abstractmethod
    def pre_process_extract_video(self, frames):
        pass

    @abstractmethod
    def train_extract(self, features):
        pass

    @abstractmethod
    def live_extract(self, frame):
        pass


class MovenetExtractor(ExtractorAbstract):
    num_features = 6 * 56
    name = "MovenetExtractor"

    def __init__(self, threshold=0.0):
        super(MovenetExtractor, self).__init__(MovenetExtractor.num_features, MovenetExtractor.name)

        # get movenet
        self.model = tf.saved_model.load('../movenet/movenet_multipose_lightning_1')  # keep ref to model
        self.movenet = self.model.signatures['serving_default']

        # threshold when outputting features
        self.threshold = threshold

    def extract_frame(self, frame):
        t1 = tf.image.resize_with_pad(frame, 256, 256)  # resize and pad
        t2 = tf.cast(t1, dtype=tf.int32)  # cast to int32
        return self.movenet(t2)['output_0']  # get result

    def pre_process_extract_video(self, frames):
        return tf.map_fn(self.extract_frame, tf.expand_dims(frames, axis=1),
                         fn_output_signature=tf.TensorSpec((1, 6, 56)),
                         )

    def train_extract(self, features):
        # zero out vals below threshold
        cond = tf.less(features, tf.constant(self.threshold, dtype=features.dtype))
        out = tf.where(cond, tf.zeros(tf.shape(features), dtype=features.dtype), features)
        return tf.reshape(out, (1, -1))  # flatten

    def live_extract(self, frame):
        t1 = self.extract_frame(frame)
        t2 = self.train_extract(t1)
        return t2


class InceptionExtractor(ExtractorAbstract):
    num_features = 2048
    name = "InceptionExtractor"

    def __init__(self, img_size):
        super(InceptionExtractor, self).__init__(InceptionExtractor.num_features, InceptionExtractor.name)
        feature_extractor = tf.keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(img_size, img_size, 3),
        )
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input

        inputs = tf.keras.Input((img_size, img_size, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        self.model = tf.keras.Model(inputs, outputs)

    def pre_process_extract_video(self, frames):
        return self.model(frames)

    def train_extract(self, features):
        return features


class MobileNetV2Extractor(ExtractorAbstract):
    num_features = 1280
    name = "MobileNetV2Extractor"

    def __init__(self, img_size=224):
        super(MobileNetV2Extractor, self).__init__(MobileNetV2Extractor.num_features, MobileNetV2Extractor.name)
        self.feature_extractor = tf.keras.applications.MobileNetV2(
            # (img_size, img_size, 3),
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )

    def pre_process_extract_video(self, frames, batch_size=32):
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

    def train_extract(self, features):
        return features

    def live_extract(self, frame):
        pre = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(frame, dtype='float32'))
        out = self.feature_extractor(pre)


if __name__ == '__main__':
    img = tf.io.decode_jpeg(tf.io.read_file('../TFlite-NN/images/image.jpeg'))
    img = tf.expand_dims(img, axis=0)
    movenet = MovenetExtractor()
    res = movenet.pre_process_extract_video(img)
    print(res.shape)
