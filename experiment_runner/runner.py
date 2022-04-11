import os
from datetime import datetime
import experiment_config as config

import tensorflow as tf
from absl import logging
logging.set_verbosity(logging.ERROR)

def train_model(exp):
    # dirs
    experiment_dir = os.path.join(f'../saved_experiments/{config.EXPERIMENT_NAME}')

    log_dir = os.path.join(experiment_dir, 'logs/fit/',
                           exp.name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # callbacks
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(experiment_dir, exp.name),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        options=None,)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch',)

    # train
    exp.model.fit(
        exp.dataset.train_dataset,
        validation_data=exp.dataset.validation_dataset,
        epochs=exp.epochs,
        batch_size=exp.batch_size,
        callbacks=[tensorboard_callback, save_model_callback],
    )

    out = exp.model.evaluate(
        exp.dataset.test_dataset
    )

    return out


if __name__ == '__main__':
    models = []
    # train test loop
    for exp in config.EXPERIMENTS:
        # init based on hyper parameters
        exp.initialize_experiment()

        # train model
        train_model(exp)
