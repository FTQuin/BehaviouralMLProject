import gc
import os
import time
from datetime import datetime
import experiment_config as config
import tensorflow as tf
from absl import logging

logging.set_verbosity(logging.ERROR)
# dirs
EXPERIMENT_DIR = f'../saved_experiments/{config.EXPERIMENT_NAME}'

def train_model(exp):
    log_dir = os.path.join(EXPERIMENT_DIR, 'logs/',
                           exp.name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # callbacks
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        f'{EXPERIMENT_DIR}/{exp.name}',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        options=None, )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch', )

    # train
    exp.model.fit(
        exp.dataset.train_dataset,
        validation_data=exp.dataset.validation_dataset,
        epochs=exp.epochs,
        batch_size=exp.batch_size,
        callbacks=[tensorboard_callback, save_model_callback],
    )

    # evaluate
    print('==== EVAL ====')
    res = exp.model.evaluate(
        exp.dataset.test_dataset
    )

    # write eval results to tensorboard
    test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'eval'))
    with test_summary_writer.as_default():
        tf.summary.scalar('test_loss', res[0], step=0)
        tf.summary.scalar('test_accuracy', res[1], step=0)

    return res


if __name__ == '__main__':
    models = []

    # train test loop
    for idx, exp in enumerate(config.EXPERIMENTS):
        # check if experiment already exists
        try:
            if exp.name in os.listdir(EXPERIMENT_DIR):
                print(f'==== Experiment {exp.name} already happened, SKIPPING ====')
                continue
        except:
            pass

        # logging
        now = time.time()
        print(f'\n\n\n==== Starting experiment {idx + 1} of {len(config.EXPERIMENTS)} ====\n')
        print('Parameters:')
        for k, v in exp.__dict__.items():
            print(f'{k}: {v}')

        # init based on hyper parameters
        exp.initialize_experiment()

        # train model
        eval_metrics = train_model(exp)

        # logging
        train_time = time.time() - now
        path_to_write = os.path.join(f'../saved_experiments/{config.EXPERIMENT_NAME}/TxtFiles')
        try:
            os.makedirs(path_to_write)
        except FileExistsError:
            pass
        with open(f'{path_to_write}/{exp.name}.txt', 'a') as out:
            for k, v in exp.__dict__.items():
                out.write(f'{k}: {v}\n')
            out.write(f'\n{eval_metrics=}\n')
            out.write(f'{train_time=}\n')
            out.write(f'\n===================\n\n')

        print(f'\n{eval_metrics=}')
        print(f'Training took {train_time} seconds')

        # memory salvage
        tf.keras.backend.clear_session()
        del exp.model
        del exp.dataset
        del exp.extractor
        del exp
        tf.keras.backend.clear_session()
        gc.collect(generation=0)
        gc.collect(generation=1)
        gc.collect(generation=2)