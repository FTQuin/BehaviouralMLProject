import tensorflow as tf
from abc import abstractmethod


class ExtractorAbstract:
    def __init__(self, num_features, name):
        self.num_features = num_features
        self.name = name

    @abstractmethod
    def pre_process_extract(self, frames):
        pass

    @abstractmethod
    def train_extract(self, features):
        pass

    def live_extract(self, frames):
        raw_features = self.pre_process_extract(frames)
        processed_features = self.train_extract(raw_features)
        return processed_features


class MovenetExtractor(ExtractorAbstract):
    def __init__(self, threshold=0.0):
        num_features = 6 * 56
        name = "MovenetExtractor"
        super(MovenetExtractor, self).__init__(num_features, name)

        # get movenet
        self.model = tf.saved_model.load('../movenet/movenet_multipose_lightning_1')  # keep ref to model
        self.movenet = self.model.signatures['serving_default']

        # threshold when outputting features
        self.threshold = threshold

    @tf.function(input_signature=(tf.TensorSpec(shape=[1, None, None, 3], dtype='int32'),))
    def pre_process_extract(self, frames):
        t1 = tf.image.resize_with_pad(frames, 256, 256)  # resize and pad
        t2 = tf.cast(t1, dtype=tf.int32)  # cast to int32
        out = self.movenet(t2)['output_0']  # get result
        return out

    @tf.function(input_signature=(tf.TensorSpec(shape=[1, 6, 56], dtype='float32'),))
    def train_extract(self, features):
        # zero out vals below threshold
        cond = tf.less(features, tf.constant(self.threshold, dtype=features.dtype))
        out = tf.where(cond, tf.zeros(tf.shape(features), dtype=features.dtype), features)
        return tf.reshape(out, (1, -1))  # flatten


class InceptionExtractor(ExtractorAbstract):
    def __init__(self, img_size):
        num_features = 2048
        name = "InceptionExtractor"
        super(InceptionExtractor, self).__init__(num_features, name)
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

    def pre_process_extract(self, frames):
        return self.model(frames)

    def train_extract(self, features):
        return features


class MobileNetV2Extractor(ExtractorAbstract):
    def __init__(self, img_size):
        num_features = 1280
        name = "MobileNetV2Extractor"
        super(MobileNetV2Extractor, self).__init__(num_features, name)
        feature_extractor = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(img_size, img_size, 3),
        )
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        inputs = tf.keras.Input((img_size, img_size, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        self.model = tf.keras.Model(inputs, outputs)

    def pre_process_extract(self, frames):
        return self.model(frames)

    def train_extract(self, features):
        return features


if __name__ == '__main__':
    img = tf.io.decode_jpeg(tf.io.read_file('../TFlite-NN/images/image.jpeg'))
    img = tf.expand_dims(img, axis=0)
    movenet = MovenetExtractor()
    res = movenet.pre_process_extract(img)
    print(res.shape)
