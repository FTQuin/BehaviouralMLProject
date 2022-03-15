import models_config as models
import preprocess_data as datasets

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
DATASET_AND_EXTRACTOR = [(datasets.UCF.movenet_extractor, {'threshold': 0.5}),
                         (datasets.NTU.movenet_extractor, {'threshold': 0.5}),
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


def train_model(model, data, params):
    return model.fit(
        data.train_data,
        data.train_labels,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
    )


def test_model(model, data, params):
    return model.evaluate(
        data.test_data,
        data.test_labels,
        batch_size=params['batch_size'],
    )


def save_model(model):
    newModel = data.featuresextractor+model
    tf.save(newModel)


def save_results(models):
    pass


if __name__ == '__main__':
    models = []
    # train test loop
    for params, data, model in zip(EXPERIMENT_PARAMS, DATASET_AND_EXTRACTOR, MODEL_PARAMS):
        model = model[0](**model[1])  # get model
        train_model(model, data, params)  # train model
        test_model(model, data, params)  # evaluate model
        save_model(models)  # save model
        models.append(model)

    save_results(models)  # save results
