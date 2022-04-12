import os
import time
from datetime import datetime
import experiment_config as config
import tensorflow as tf
from absl import logging

logging.set_verbosity(logging.ERROR)


def train_model(exp):
    # dirs
    experiment_dir = os.path.join(f'../saved_experiments/{config.EXPERIMENT_NAME}')

    log_dir = os.path.join(experiment_dir, 'logs/',
                           exp.name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # callbacks
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(f'{experiment_dir}/{exp.name}'),
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
    print('==== EVAL ===')
    out = exp.model.evaluate(
        exp.dataset.test_dataset
    )

    return out


if __name__ == '__main__':
    models = []

    # train test loop
    for idx, exp in enumerate(config.EXPERIMENTS):
        now = time.time()
        try:
            os.makedirs(f'../saved_experiments/{exp.extractor.name}TxtFiles')
        except:
            pass
        path_to_write = os.path.join(f'../saved_experiments/{exp.extractor.name}TxtFiles')
        print(f'\n\n\n==== Starting experiment {idx + 1} of {len(config.EXPERIMENTS)} ====\n')
        print('Parameters:')
        for k, v in exp.__dict__.items():
            with open(f'{path_to_write}/{exp.name}.txt', 'a') as out:
                print(f'{k}: {v}')
                out.write(f'{k}: {v}\n')



        # init based on hyper parameters
        exp.initialize_experiment()

        # train model
        eval_loss_acc = train_model(exp)
        with open(f'{path_to_write}/{exp.name}.txt', 'a') as out:
            out.write(f'\n{eval_loss_acc=}\n')

        print(f'\n{eval_loss_acc=}')
        print(f'Training took {time.time() - now} seconds')
        tf.keras.backend.clear_session()
