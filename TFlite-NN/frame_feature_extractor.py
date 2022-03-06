"""
:Date: 2022-02-28
:Author: Quin Adam
:Description:
    This module was created for feature extraction from videos.
    It is designed to take in a frame, and return the skeletons
    of the people in the frame. The skeletons map out both joints
    and bones, and are used as features for machine learning.
"""
import math
import tensorflow as tf


def downsize_frame(frame, target_size=256):
    """
    Downsizes an image to the target_size on its longest axis.

    :param frame: a [1, h, w, c] image, h: height, w: width, c: channel (RGB)
    :type frame: [1, h, w, c] image
    :param target_size: a target length for the long side of the image
    :type target_size: int
    :return: an image tensor who's longest size is target size
    :rtype: [1, h, w, c] tensor
    """
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
interpreter = tf.lite.Interpreter(model_path='./movenet/lite-model_movenet_multipose_lightning_tflite_float16_1.tflite')
interpreter.allocate_tensors()
def movenet(scaled_frame_tensor):
    """
    Takes an image and returns the joints of skeletons in the image.
    The longest side of the image must be 256 pixels.

    :param scaled_frame_tensor: a [1, h, w, c] image who's longest side is 256 pixels,
        h: height, w: width, c: channel (RGB)
    :type scaled_frame_tensor: [1, h, w, c] tensor
    :return: a [1, 6, 56] ndarray, the second dimension is for each skeleton, the third dimension has multiple parts:
        the first 17*3=51 floats are (y, x, confidence-score) of each of the 17 joints, the last 5 are (y-min, x-min,
        y-max, x-max, confidence-score) of the bounding box around the skeleton
    :rtype: [1, 6, 56] ndarray
    """

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
    """
    Changes raw movenet data to a more usable format. Specifically it puts
    joints and bounding box in their own tensors. Also returns a mask of which
    skeletons are viable based on the bounding box confidence score.

    :param movenet_result: an ndarray containing the result from the movenet interpreter
    :type movenet_result: [1, 6, 56] ndarray
    :return: **joints**: a tensor with the locations of the joints of each skeleton,
        **boxes**: a tensor with the bounds of each skeleton,
        **people_mask**: a mask of which skeletons are actually detected
    :rtype: ([1, 6, 17, 3] tensor, [1, 6, 4] tensor, [1, 6, 1] tensor)
    """

    # split raw tensor
    joints, boxes, confidence = tf.split(movenet_result, [56-4-1, 4, 1], 2)

    # split joints
    joints = tf.split(joints[0], num_or_size_splits=17, axis=1)
    joints = tf.convert_to_tensor(joints, dtype=tf.float32)
    joints = tf.transpose(joints, perm=[1, 0, 2])
    joints = tf.expand_dims(joints, axis=0)

    #TODO: turn confidence into a mask based on a threshold hyperparameter
    return joints, boxes, confidence


# TODO: this
def remove_unusable_joints(joints):
    """
    Creates a mask of joints with a low confidence score. Also combines the 5 joints on the head into just one.

    :param joints: a tensor with the locations of the joints of each skeleton
    :type joints: [1, 6, 17, 3] tensor
    :return: **joints**: the same input joints tensor with the head joints combined,
        **joint_mask**: a mask for the joints based on the confidence score of them
    :rtype: ([1, 6, 13, 3] tensor, [1, 6, 13] tensor)
    """
    pass


# TODO: this
def create_bone_feature(joints):
    """
    Uses joints to create features from their connections (bones)

    :param joints: a tensor with the locations of the joints of each skeleton
    :type joints: [1, 6, 13, 3] tensor
    :return: a tensor with the angle and length of each bone
    :rtype: [1, 6, 14, 2] tensor
    """
    pass


# TODO: this
def combine_feature(joints, bones):
    """
    Combines the features from joints and bones into one feature tensor for ML training and inference.

    :param joints: a tensor with the locations of the joints of each skeleton
    :type joints: [1, 6, 13, 3] tensor
    :param bones: a tensor with the angle and length of each bone
    :type bones: [1, 6, 14, 2] tensor
    :return: a tensor with these features combined into one feature tensor, [1, 6, (13*3)+(14*2)=67]
    :rtype: [1, 6, 67] tensor
    """
    pass


def get_features_from_image(frame):
    """
    Takes in an image of any size and extracts the skeletal features of that image.

    :param frame: a [1, h, w, c] image, h: height, w: width, c: channel (RGB)
    :type frame: [1, h, w, c] image
    :return: a tensor with the extracted features
    :rtype: [1, 6, 17, 3] tensor
    """

    # add extra dim to image
    frame = tf.expand_dims(frame, axis=0)
    # downsize image
    downsized_image = downsize_frame(frame)
    # run through movenet
    raw_joints = movenet(downsized_image)
    # convert joints to more readable
    joints, boxes, confidence = reorganize_movenet_result(raw_joints)

    return joints


if __name__ == '__main__':
    # get test image
    image_path = 'images/image.jpeg'
    image = tf.io.read_file(image_path)
    image = tf.compat.v2.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)

    # downsize image
    downsized_image = downsize_frame(image, 256)

    # run through movenet
    raw_joints = movenet(downsized_image)

    # convert joints to more usable format
    joints, boxes, confidence = reorganize_movenet_result(raw_joints)

    print(joints)


