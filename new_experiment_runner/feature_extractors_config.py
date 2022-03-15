import tensorflow as tf


class movenet_extractor():
    def __init__(self, name=None):
        super(movenet_extractor, self).__init__()

        model = tf.saved_model.load('../movenet/movenet_multipose_lightning_1')
        self.movenet = model.signatures['serving_default']

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype='uint8'),))
    def extract(self, input):
        t1 = tf.image.resize_with_pad(input, 256, 256)  # resize and pad
        t2 = tf.cast(t1, dtype=tf.int32)  # cast to int32
        out = self.movenet(t2)['output_0']  # get result
        return out


if __name__ == '__main__':
    img = tf.io.decode_jpeg(tf.io.read_file('../TFlite-NN/images/image.jpeg'))
    img = tf.expand_dims(img, axis=0)
    movenet = movenet_extractor()
    res = movenet.extract(img)
    print(res.shape)
