import datetime

import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

from tensorflow_docs.vis import embed
from tensorflow import keras


def detect(interpreter, input_tensor):
    """Runs detection on an input image.

    Args:
    interpreter: tf.lite.Interpreter
    input_tensor: A [1, input_height, input_width, 3] Tensor of type tf.float32.
      input_size is specified when converting the model to TFLite.

    Returns:
    A tensor of shape [1, 6, 56].
    """

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
    if is_dynamic_shape_model:
        input_tensor_index = input_details[0]['index']
        input_shape = input_tensor.shape
        interpreter.resize_tensor_input(
            input_tensor_index, input_shape, strict=True)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

    interpreter.invoke()

    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


def keep_aspect_ratio_resizer(image, target_size):
    """Resizes the image.

    The function resizes the image such that its longer side matches the required
    target_size while keeping the image aspect ratio. Note that the resizes image
    is padded such that both height and width are a multiple of 32, which is
    required by the model.
    """
    _, height, width, _ = image.shape
    if height > width:
        scale = float(target_size / height)
        target_height = target_size
        scaled_width = math.ceil(width * scale)
        image = tf.image.resize(image, [target_height, scaled_width])
        target_width = int(math.ceil(scaled_width / 32) * 32)
    else:
        scale = float(target_size / width)
        target_width = target_size
        scaled_height = math.ceil(height * scale)
        image = tf.image.resize(image, [scaled_height, target_width])
        target_height = int(math.ceil(scaled_height / 32) * 32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
    return (image, (target_height, target_width))


# Load the input image.
input_size = 256
image_path = 'image.jpeg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)

# Resize and pad the image to keep the aspect ratio and fit the expected size.
resized_image, image_shape = keep_aspect_ratio_resizer(image, input_size)
image_tensor = tf.cast(resized_image, dtype=tf.uint8)

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_multipose_lightning_tflite_float16_1.tflite')

# Output: [1, 6, 56] tensor that contains keypoints/bbox/scores.
keypoints_with_scores = detect(
    interpreter, tf.cast(image_tensor, dtype=tf.uint8))

"""
## Define hyperparameters
"""

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 56 * 6

train_df = pd.read_csv("../First NN/train.csv")
test_df = pd.read_csv("../First NN/test.csv")


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        scale_constant = 255 / (255 ** 2.2)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frame = frame.astype('float64') ** 2.2 * scale_constant
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        classifier_activation='softmax',
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return detect


feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        image = tf.expand_dims(frames, axis=0)

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        resized_image, image_shape = keep_aspect_ratio_resizer(image, input_size)

        # if idx > 1:
        #     break
        print("Extracting features of video: ", idx)

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(image):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                image_tensor = tf.cast(resized_image[j], dtype=tf.uint8)
                temp_frame_features[i, j] = detect(
                    interpreter, tf.cast(image_tensor, dtype=tf.uint8)).flatten()
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


# prepare data
train_data, train_labels = prepare_all_videos(train_df, "../First NN/train")
test_data, test_labels = prepare_all_videos(test_df, "../First NN/test")

# # save data
np.savez("lite_data", train_data_0=train_data[0], train_data_1=train_data[1], train_labels=train_labels,
         test_data_0=test_data[0], test_data_1=test_data[1], test_labels=test_labels)
print("dumped")

# load data
train_data_0, train_data_1, train_labels, test_data_0, test_data_1, test_labels \
    = [item[1] for item in np.load("lite_data.npz").items()]
train_data = (train_data_0, train_data_1)
test_data = (test_data_0, test_data_1)
print("loaded")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

"""
The above code block will take ~20 minutes to execute depending on the machine it's being
executed.
"""


# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(2000, activation='relu')(x)
    x = keras.layers.Dense(1000, activation='relu')(x)
    x = keras.layers.Dense(200, activation='relu')(x)
    output = keras.layers.Dense(len(class_vocab), activation='softmax')(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# Utility for running experiments.
def run_experiment():
    filepath = "First NN/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=False, verbose=1
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint, tensorboard_callback],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model, accuracy


_, sequence_model, test_accuracy = run_experiment()

"""
Test
"""


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("../First NN/test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("../First NN/animation.gif")


test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])


def frames_to_prediction(frames):
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    class_vocab = label_processor.get_vocabulary()

    probs = np.argsort(probabilities)[::-1]
    for i in probs:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return probs