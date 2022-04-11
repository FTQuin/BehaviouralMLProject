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


EXPERIMENT_NAME = 'TEST'
e = Experiment(name='test', batch_size=16, epochs=5,
               dataset_path='../datasets/NTU-3', dataset_params={'seq_len': 20,
                                                                 'train_test_split': .8},
               extractor=feature_extractors.MovenetExtractor,
               model=models.GRU.gru1, model_params={'activation_function': 'relu',
                                                    'loss_function': 'sparse_categorical_crossentropy',
                                                    'optimizer': 'adam'}
               )

EXPERIMENTS = [
    e,
    e.like(name='not rest')
]