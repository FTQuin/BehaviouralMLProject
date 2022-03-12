import define_models as models
import preprocess_data as datasets

# EXPERIMENT
EXPERIMENT_NAME = 'test_experiment'
EXPERIMENT_PARAMS = [
                     {'batch_size': 32,
                      'epochs': 10,
                      },
                     {'batch_size': 32,
                      'epochs': 10,
                      },
                     ]

# DATA
FEATURE_DATA = [
                datasets.UCF.movenet_extractor,
                datasets.NTU.movenet_extractor,
                ]

# MODELS
TRAIN_NETWORKS = True
MODEL_PARAMS = [(models.rnn, {'activation_function': 'relu',
                              'loss_function': 'sparse_categorical_crossentropy',
                              'optimizer': 'adam',
                              }),
                (models.rnn, {'activation_function': 'relu',
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


def test_model():
    return model.evaluate(
        data.test_data,
        data.test_labels,
        batch_size=params['batch_size'],
    )


def save_results(model):
    pass


def save_models(models):
    pass


if __name__ == '__main__':
    models = []
    # train test loop
    for params, data, model in zip(EXPERIMENT_PARAMS, FEATURE_DATA, MODEL_PARAMS):
        model = model[0](**model[1])  # get model
        train_model(model, data, params)  # train model
        test_model(model, data, params)  # evaluate model
        save_results(model)  # save results
        models.append(model)

    save_models(models)  # save all models
