import math
import tensorflow as tf


def downsize_frame(frame, target_size=256):
    # takes in an image and a target for the long side of the image
    # returns and image who's longest size is target size

    _, height, width, _ = frame.shape
    if height > width:
        scale = float(target_size / height)
        target_height = target_size
        scaled_width = math.ceil(width * scale)
        frame = tf.image.resize(frame, [target_height, scaled_width])
        target_width = int(math.ceil(scaled_width / 32) * 32)
    else:
        scale = float(target_size / width)
        target_width = target_size
        scaled_height = math.ceil(height * scale)
        frame = tf.image.resize(frame, [scaled_height, target_width])
        target_height = int(math.ceil(scaled_height / 32) * 32)

    frame = tf.image.pad_to_bounding_box(frame, 0, 0, target_height, target_width)
    return frame


# init movenet
interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_multipose_lightning_tflite_float16_1.tflite')
interpreter.allocate_tensors()
def movenet(scaled_frame_tensor):
    # takes in an image tensor
    # returns as 1x6x56 tensor of features

    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(scaled_frame_tensor, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
    if is_dynamic_shape_model:
        input_tensor_index = input_details[0]['index']
        input_shape = input_image.shape
        interpreter.resize_tensor_input(
            input_tensor_index, input_shape, strict=True)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()

    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    return keypoints_with_scores


def reorganize_movenet_result(movenet_result):
    # takes in 1x6x56 tensor
    # returns
    # 1x6x17x3 tensor joints,
    # 1x6x4 tensor bounding_boxes,
    # 1x6x1 people_mask

    # split raw tensor
    joints, boxes, confidence = tf.split(movenet_result, [56-4-1, 4, 1], 2)

    # split joints
    joints = tf.split(joints[0], num_or_size_splits=17, axis=1)
    joints = tf.convert_to_tensor(joints, dtype=tf.float32)
    joints = tf.transpose(joints, perm=[1, 0, 2])
    joints = tf.expand_dims(joints, axis=0)

    return joints, boxes, confidence


def remove_unusable_joints(movenet_result):
    pass


def create_bone_feature(joints):
    pass


def combine_feature(joints, bones):
    pass


# TODO: complete rest of functions
def get_features_from_image(image):
    # takes in an image [w, h, c]
    # returns feature vector 1x6x17x3

    # add extra dim to image
    image = tf.expand_dims(image, axis=0)
    # downsize image
    downsized_image = downsize_frame(image)
    # run through movenet
    raw_joints = movenet(downsized_image)
    # convert joints to more readable
    joints, boxes, confidence = reorganize_movenet_result(raw_joints)

    return joints


if __name__ == '__main__':
    # get test image
    image_path = 'image.jpeg'
    image = tf.io.read_file(image_path)
    image = tf.compat.v2.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)

    # downsize image
    downsized_image = downsize_frame(image, 256)

    # run through movenet
    raw_joints = movenet(downsized_image)

    # convert joints to more readable
    joints, boxes, confidence = reorganize_movenet_result(raw_joints)

    print(joints)


