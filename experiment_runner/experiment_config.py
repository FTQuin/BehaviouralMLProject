import os.path
from dataclasses import dataclass
from typing import Callable

import models_config as models
import datasets_config as datasets
import feature_extractors_config as feature_extractors

@dataclass
class Experiment:
    name: str
    batch_size: int
    epochs: int
    dataset_path: str
    dataset_params: dict
    extractor: Callable
    model: Callable
    model_params: dict
    dataset: datasets.Dataset.Training = None

    def like(self, **kwargs):
        params = self.__dict__.copy()
        params.update(kwargs)
        return Experiment(**params)

    def initialize_experiment(self):
        self.extractor = self.extractor()  # get extractor
        self.dataset = datasets.Dataset.Training(self.dataset_path, **self.dataset_params, extractor=self.extractor)  # get dataset
        self.model = self.model(output_size=len(self.dataset.labels), **self.model_params)  # get model

    def gen_name(self):
        s = f'{os.path.split(self.dataset_path)[-1]}_{self.extractor.name}_{self.model_params}_{self.dataset_params}_{self.model.__name__}'
        return s.translate({ord(i): '' for i in '\':{}[],.'}).translate({ord(i): '_' for i in ' '})


FEATURE_EXTRACTORS = {'extractor': [feature_extractors.MovenetExtractor, feature_extractors.InceptionExtractor, feature_extractors.MobileNetV2Extractor]}
MODEL_PARAMS = {'model_params': [{'activation_function': 'relu', 'optimizer': 'adam'},
                                 {'activation_function': 'tanh', 'optimizer': 'adam'},
                                 {'activation_function': 'sigmoid', 'optimizer': 'adam'},
                                 {'activation_function': 'relu', 'optimizer': 'adagrad'},
                                 {'activation_function': 'tanh', 'optimizer': 'adagrad'},
                                 {'activation_function': 'sigmoid', 'optimizer': 'adagrad'}
                                 ]}
DATASET_PARAMS = {'dataset_params': [{'seq_len': 10, 'train_test_split': .80, 'enable_caching': True},
                                     {'seq_len': 20, 'train_test_split': .80, 'enable_caching': True},
                                     {'seq_len': 50, 'train_test_split': .80, 'enable_caching': True}
                                     ]}

MODELS = {'model': [models.GRU.gru1, models.GRU.gru2, models.LSTM.lstm1, models.LSTM.lstm2]}

e = Experiment(
    name='test', batch_size=32, epochs=10,
    dataset_path='../datasets/UCF-3', dataset_params={'seq_len': 20,
                                                      'train_test_split': .8},
    extractor=feature_extractors.MovenetExtractor,
    model=models.GRU.gru1, model_params={'activation_function': 'relu',
                                         'optimizer': 'adam'}
)

EXPERIMENTS = [e]

def grid(exp_list, param_list):
    l = []
    for exp in exp_list:
        for p in list(param_list.values())[0]:
            d = {list(param_list.keys())[0]: p}
            temp = exp.like(**d)
            l.append(temp.like(name=str(temp.gen_name())))
    return l

EXPERIMENTS = grid(EXPERIMENTS, FEATURE_EXTRACTORS)
EXPERIMENTS = grid(EXPERIMENTS, MODEL_PARAMS)
EXPERIMENTS = grid(EXPERIMENTS, DATASET_PARAMS)
EXPERIMENTS = grid(EXPERIMENTS, MODELS)


EXPERIMENT_NAME = 'GRID'

