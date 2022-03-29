import os
from datetime import datetime

import tensorflow as tf

import models_config as models
import datasets_config as datasets
import feature_extractors_config as feature_extractors

# EXPERIMENT
EXPERIMENT_NAME = 'mobilenet_experiments_1'
EXPERIMENT_PARAMS = [
    {'name': 'gru',
     'batch_size': 64,
     'epochs': 5,
     },
    {'name': 'lstm',
     'batch_size': 64,
     'epochs': 5,
     },
]

# DATA
DATASETS_PARAMS = [
    {'dataset_path': '../datasets/UCF-3',
     'seq_len': 50,
     'train_test_split': .8
     },
    {'dataset_path': '../datasets/UCF-3',
     'seq_len': 50,
     'train_test_split': .8
     },
]

# EXTRACTOR
EXTRACTOR_PARAMS = [
    (feature_extractors.MobileNetV2Extractor, {}),
    (feature_extractors.MobileNetV2Extractor, {}),
]

# MODELS
MODEL_PARAMS = [
    (models.GRU.gru2, {'activation_function': 'relu',
                       'loss_function': 'sparse_categorical_crossentropy',
                       'optimizer': 'adam',
                       }),
    (models.LSTM.lstm2, {'activation_function': 'relu',
                         'loss_function': 'sparse_categorical_crossentropy',
                         'optimizer': 'adam',
                         }),
]


def train_model(model, dataset, experiment_params):
    experiment_dir = os.path.join('../saved_experiments', EXPERIMENT_NAME)

    log_dir = os.path.join(experiment_dir, 'logs/fit/',
                           experiment_params['name'] + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    save_model_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(experiment_dir, experiment_params['name']),
                                                             monitor='val_loss',
                                                             verbose=1,
                                                             save_best_only=True,
                                                             options=None,)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          update_freq='epoch',)

    out = model.fit(
        dataset.train_dataset,
        validation_data=dataset.validation_dataset,
        epochs=experiment_params['epochs'],
        batch_size=experiment_params['batch_size'],
        callbacks=[tensorboard_callback, save_model_callback],
    )
    return out


if __name__ == '__main__':
    # train test loop
    for idx, (experiment_params, data_params, extractor_params, model_params) in \
            enumerate(zip(EXPERIMENT_PARAMS, DATASETS_PARAMS, EXTRACTOR_PARAMS, MODEL_PARAMS)):

        # init based on hyper parameters
        extractor = extractor_params[0](**extractor_params[1])  # get extractor
        dataset = datasets.Dataset.Training(**data_params, extractor=extractor)  # get data
        model = model_params[0](output_size=len(dataset.labels), **model_params[1])  # get model

        train_model(model, dataset, experiment_params)  # train model
        # test_model(model, dataset, experiment_params)  # evaluate model
