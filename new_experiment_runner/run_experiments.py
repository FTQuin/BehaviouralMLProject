import define_models as models
import preprocess_data as datasets

# EXPERIMENT
EXPERIMENT_NAME = 'test_experiment'
EXPERIMENT_PARAMS = [{'validation_split': 0.3,
                      'epochs': 10,
                      },
                     {'validation_split': 0.3,
                      'epochs': 10,
                      },
                     ]

# DATA
FEATURE_DATA = [(datasets.UCF.movenet_extractor, {'train_test_split': 0.3,
                                                  }),
                (datasets.NTU.movenet_extractor, {'train_test_split': 0.3,
                                                  }),
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



def get_data():
    pass


def get_model():
    pass


def train_model():
    pass


def save_results():
    pass


if __name__ == '__main__':
    for params, data, model in zip(EXPERIMENT_PARAMS, FEATURE_DATA, MODEL_PARAMS):
