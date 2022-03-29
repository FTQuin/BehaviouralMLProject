import os
from datetime import datetime

import tensorflow as tf

import models_config as models
import datasets_config as datasets
import feature_extractors_config as feature_extractors
# import tensorflow as tf

# EXPERIMENT
EXPERIMENT_NAME = 'mobilenet_experiments_1'
EXPERIMENT_PARAMS = [{'name': 'gru',
                      'batch_size': 64,
                      'epochs': 5,
                      },
                     {'name': 'lstm',
                      'batch_size': 64,
                      'epochs': 5,
                      },
                     ]

# DATA
DATASETS_PARAMS = [(datasets.UCF.training, {'seq_len': 50, 'train_test_split': .8}),
                   (datasets.UCF.training, {'seq_len': 50, 'train_test_split': .8}),
                   ]

# EXTRACTOR
EXTRACTOR_PARAMS = [(feature_extractors.MobileNetV2Extractor, {}),
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


def train_model(model, dataset, experiment_params, idx):
    # train_iter = dataset.get_train_data(MODEL_PARAMS[idx][1]['seq_len'])

    log_dir = os.path.join('../saved_experiments',
                           EXPERIMENT_NAME,
                           'logs/fit/',
                           experiment_params['name'] + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          update_freq='epoch',)

    out = model.fit(
        dataset.train_dataset,
        validation_data=dataset.dataset_validation,
        epochs=experiment_params['epochs'],
        batch_size=experiment_params['batch_size'],
        callbacks=[tensorboard_callback],
    )
    return out


def test_model(model, dataset, experiment_params):
    x = dataset.get_test_data(MODEL_PARAMS[idx][1]['seq_len'])
    y = dataset.get_test_labels()

    dir_path = os.path.join('../saved_experiments', EXPERIMENT_NAME, 'logs/eval/')
    logdir = dir_path + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                          histogram_freq=1,
                                                          profile_batch='500,520')

    out = model.evaluate(
        x, y,
        batch_size=experiment_params['batch_size'],
        callbacks=[tensorboard_callback]
    )
    return out


def save_model(model, extractor, experiment_params, idx):
    dir_path = os.path.join('../saved_experiments', EXPERIMENT_NAME, str(idx)+'_'+experiment_params['name'])
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
            t1 = self.extractor.live_extract(x)
            t2 = tf.random.uniform((1, 20-1, 6*56))
            t3 = tf.concat([t2, tf.reshape(t1, (1, -1))[None, :]], 1)
            out = self.model(t3, training=False)
            return out

    # total_mod = TotalModel()
    # call_output = total_mod.__call__.get_concrete_function(tf.random.uniform((1, 1000, 1000, 3), dtype='int32', maxval=255))
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
        model = model_params[0](output_size=len(dataset.labels), **model_params[1])  # get model

        train_model(model, dataset, experiment_params, idx)  # train model
        # test_model(model, dataset, experiment_params)  # evaluate model

        save_model(model, extractor, experiment_params, idx)  # save model
        models.append(model)

    save_results(models)  # save results
