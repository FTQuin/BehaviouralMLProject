import os
from datetime import datetime

import numpy as np
import tensorflow as tf

import models_config as models
import datasets_config as datasets
import feature_extractors_config as feature_extractors
# import tensorflow as tf

# EXPERIMENT
EXPERIMENT_NAME = 'test_experiment_3'
EXPERIMENT_PARAMS = [{'batch_size': 16,
                      'epochs': 25,
                      },
                     {'batch_size': 32,
                      'epochs': 20,
                      }
                     ]

# DATA
DATASETS_PARAMS = [(datasets.UCF, {'train_test_split': .75}),
                   (datasets.UCF, {'train_test_split': .75}),
                   ]

# EXTRACTOR
EXTRACTOR_PARAMS = [(feature_extractors.MovenetExtractor, {'threshold': 0.5}),
                    (feature_extractors.MovenetExtractor, {'threshold': 0.5}),
                    ]

# MODELS
MODEL_PARAMS = [(models.GRU.gru2, {'seq_len': 20,
                                   'activation_function': 'relu',
                                   'loss_function': 'sparse_categorical_crossentropy',
                                   'optimizer': 'adam',
                                   }),
                (models.GRU.gru2, {'seq_len': 29,
                                   'activation_function': 'sigmoid',
                                   'loss_function': 'sparse_categorical_crossentropy',
                                   'optimizer': 'adam',
                                   }),
                ]


def train_model(model, dataset, experiment_params):
    x = dataset.train_data(MODEL_PARAMS[0][1]['seq_len'])
    y = dataset.train_labels()

    dir_path = os.path.join('../saved_experiments', EXPERIMENT_NAME, 'logs/fit/')
    logdir = dir_path + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)

    out = model.fit(
        x,
        y,
        epochs=experiment_params['epochs'],
        batch_size=experiment_params['batch_size'],
        callbacks=[tensorboard_callback]
    )
    return out


def test_model(model, dataset, experiment_params):
    return model.evaluate(
        dataset.test_data,
        dataset.test_labels,
        batch_size=experiment_params['batch_size'],
    )


def save_model(model, extractor, idx):
    dir_path = os.path.join('../saved_experiments', EXPERIMENT_NAME, idx)
    try:
        os.mkdir(dir_path)
    except:
        pass

    class TotalModel(tf.Module):
        def __init__(self):
            super(TotalModel, self).__init__()
            self.extractor = extractor
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec([1, None, None, 3], tf.int32)])
        def __call__(self, x):
            t1 = self.extractor.extract(x)
            t2 = tf.random.uniform((1, 20-1, 6*56))
            t3 = tf.concat([t2, tf.reshape(t1, (1, -1))[None, :]], 1)
            out = self.model(t3, training=False)
            return out

    total_mod = TotalModel()
    call_output = total_mod.__call__.get_concrete_function(tf.random.uniform((1, 1000, 1000, 3), dtype='int32', maxval=255))
    # tf.saved_model.save(call_output, dir_path)
    tf.saved_model.save(model, dir_path)


def save_results(models):
    pass


if __name__ == '__main__':
    models = []
    # train test loop
    for idx, (experiment_params, data_params, extractor_params, model_params) in \
            enumerate(zip(EXPERIMENT_PARAMS, DATASETS_PARAMS, EXTRACTOR_PARAMS, MODEL_PARAMS)):

        # init based on hyper parameters
        extractor = extractor_params[0](**extractor_params[1])  # get extractor
        dataset = data_params[0](**data_params[1], extractor=extractor)  # get data
        model = model_params[0](input_shape=extractor.num_features, output_size=3, **model_params[1])  # get model

        train_model(model, dataset, experiment_params)  # train model
        # test_model(model, dataset, experiment_params)  # evaluate model

        save_model(model, extractor, idx)  # save model
        models.append(model)

    save_results(models)  # save results
