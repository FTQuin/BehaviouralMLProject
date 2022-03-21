import tensorflow as tf
from abc import abstractmethod


class ExtractorAbstract:
    def __init__(self, num_features):
        self.num_features = num_features
    @abstractmethod
    def pre_process_features(self, frames):
        pass

    @abstractmethod
    def post_process_features(self, features):
        pass

    def extract(self, frames):
        raw_features = self.pre_process_features(frames)
        processed_features = self.post_process_features(raw_features)
        return processed_features


class MovenetExtractor(ExtractorAbstract):
    def __init__(self, threshold=0.0):
        num_features = 6*56
        super(MovenetExtractor, self).__init__(num_features)

        # get movenet
        self.model = tf.saved_model.load('../movenet/movenet_multipose_lightning_1') # keep ref to model
        self.movenet = self.model.signatures['serving_default']

        # threshold when outputting features
        self.threshold = threshold

    @tf.function(input_signature=(tf.TensorSpec(shape=[1, None, None, 3], dtype='int32'),))
    def pre_process_features(self, frames):
        t1 = tf.image.resize_with_pad(frames, 256, 256)  # resize and pad
        t2 = tf.cast(t1, dtype=tf.int32)  # cast to int32
        out = self.movenet(t2)['output_0']  # get result
        return out

    @tf.function(input_signature=(tf.TensorSpec(shape=[1, 6, 56], dtype='float32'),))
    def post_process_features(self, features):
        cond = tf.less(features, tf.constant(self.threshold, dtype=features.dtype))
        out = tf.where(cond, tf.zeros(tf.shape(features), dtype=features.dtype), features)
        return tf.reshape(out, (1, -1))  # flatten


if __name__ == '__main__':
    img = tf.io.decode_jpeg(tf.io.read_file('../TFlite-NN/images/image.jpeg'))
    img = tf.expand_dims(img, axis=0)
    movenet = MovenetExtractor()
    res = movenet.pre_process_features(img)
    print(res.shape)
