import models_config as models
import datasets_config as datasets
import feature_extractors_config as feature_extractors

# EXPERIMENT
EXPERIMENT_NAME = 'test_experiment'
EXPERIMENT_PARAMS = [
                     {'batch_size': 16,
                      'epochs': 10,
                      },
                     {'batch_size': 32,
                      'epochs': 20,
                      },
                     ]

# DATA
DATASET_AND_EXTRACTOR = [(datasets.UCF, {'extractor': feature_extractors.MovenetExtractor,
                                         'threshold': 0.5}),
                         (datasets.NTU, {'extractor': feature_extractors.MovenetExtractor,
                                         'threshold': 0.5}),
                         ]

# MODELS
TRAIN_NETWORKS = True
MODEL_PARAMS = [(models.GRU.gru1, {'activation_function': 'relu',
                                   'loss_function': 'sparse_categorical_crossentropy',
                                   'optimizer': 'adam',
                                   }),
                (models.GRU.gru2, {'activation_function': 'sigmoid',
                                   'loss_function': 'sparse_categorical_crossentropy',
                                   'optimizer': 'adam',
                                   }),
                ]


def train_model(model, data, experiment_params):

    return model.fit(
        data.train_data,
        data.train_labels,
        epochs=experiment_params['epochs'],
        batch_size=experiment_params['batch_size'],
    )


def test_model(model, data, params):
    return model.evaluate(
        data.test_data,
        data.test_labels,
        batch_size=params['batch_size'],
    )


def save_model(model):
    newModel = data_params.featuresextractor + model
    tf.save(newModel)


def save_results(models):
    pass


def init_model(model_params):
    model = model_params[0](**model_params[1])  # get model
    return model


def init_data(data_params):
    dataset = data_params[0](**data_params[1])
    dataset =


if __name__ == '__main__':
    models = []
    # train test loop
    for experiment_params, data_params, model_params in zip(EXPERIMENT_PARAMS, DATASET_AND_EXTRACTOR, MODEL_PARAMS):
        # init based on hyper parameters
        model = init_model(model_params)
        data = init_data(data_params)

        train_model(model, data, experiment_params)  # train model
        test_model(model, data, experiment_params)  # evaluate model

        save_model(model, data, experiment_params)  # save model
        models.append(model)

    save_results(models)  # save results
