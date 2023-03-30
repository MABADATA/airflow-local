from sklearn.ensemble import (AdaBoostClassifier,BaggingClassifier,
                              GradientBoostingClassifier,RandomForestClassifier)
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer,SGD
import logging
import json
import torch.nn as nn
import collections
from Estimator_match import Estimator_handler
from Input_validation import Input_validatior
from Env_prep import Envsetter

from torch.utils.data import Dataset

from Bucket_loader import Bucket_loader
import tensorflow as tf
from keras.losses import Loss
import pandas as pd
import numpy as np
class Test:
    def __init__(self, model_type, dataloader_type,
                 ml_type='classification', loss=False,
                 input_shape=False, nb_classes=False,
                 input_val_range=False, algorithm=None,
                 optimizer=False, valid=True,
                 value_dict={'model': None, 'dataloader': None,
                             'loss': None,'input_shape': None, 'nb_classes': None,
                             'input_val_range': False, 'algorithm': None,'optimizer': False}
                 ):
        """
        The test class is a test generator based on the input values.

        parameters
        ----------
          model_type: str, kind of implementation the model is.valid inputs are only sklearn, pytorch, tensorflow.

          dataloader_type: str, type of the dataloader.

          ml_type: str, problem the ML solves. valid inputs are only classification and regression.

          loss: bool, if true a loss function will be generated.

          input_shape: bool, provide an input shape or not. If True input shape will be generated.
                        needs to be True if:
                            1.model_implementation_type == "tensorflow"
                            2.ml_type != 'regression'
                            or
                            1.model_implementation_type == "pytorch"
                            2. ml_type == 'regression'/"classification"

          nb_classes: bool, provide number of classes or not. If True number of classes will be generated.
                        needs to be True if:
                            1.model_implementation_type == "tensorflow"
                            2.ml_type != 'regression'
                            or
                            1.model_implementation_type == "pytorch"
                            2. ml_type == "classification"

          input_val_range: bool, provide an input_val_range or not. If True input_val_range will be generated.
                            needs to be True if:
                                model_implementation_type == "sklearn"

          algorithm: str, type of algorithm used. only relevant for sklearn models.

          optimizer: bool, if true an optimizer will be generated.

          valid: bool, if True, a valid data will be generated.

          value_dict: dict, a dictionary that holds values to be tested on a specified param.
                            note: if the dict has a value in some param, the argument of the param has to be True.

        """

        self.model_type = model_type
        self.dataloader_type = dataloader_type
        self.ml_type = ml_type
        self.loss = loss
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.input_val_range = input_val_range
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.valid = valid
        self.value_dict = value_dict
        self.valid_model_types = ["sklearn", "pytorch", "tensorflow"]
        self.valid_dataloader_type = ["list", "DataFrame", "ndarray", "array", "dataset"]
        self.valid_ml_type = ["regression", "classification"]
        self.valid_algorithm_type = ["DecisionTree",
                                     "ExtraTree",
                                     "AdaBoost",
                                     "Bagging",
                                     "GradientBoosting",
                                     "RandomForest",
                                     "LogisticRegression",
                                     "GaussianNB",
                                     "SVC",
                                     "LinearSVC"]

    def gen_pytorch_model(self):
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        return model

    def gen_tensorflow_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", input_shape=(28, 28, 1)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])
        return model

    def gen_sklearn_model(self, alg):
        map = {"DecisionTree": DecisionTreeClassifier,
               "ExtraTree": ExtraTreeClassifier,
               "AdaBoost": AdaBoostClassifier,
               "Bagging": BaggingClassifier,
               "GradientBoosting": GradientBoostingClassifier,
               "RandomForest": RandomForestClassifier,
               "LogisticRegression": LogisticRegression,
               "GaussianNB": GaussianNB,
               "SVC": SVC,
               "LinearSVC": LinearSVC,
               }
        model = map[alg]()
        return model

    def gen_model(self):
        if self.value_dict['model']:
            return self.value_dict['model']
        if self.model_type in self.valid_model_types:
            if self.model_type == "sklearn":
                if self.algorithm in self.valid_algorithm_type:
                    model = self.gen_sklearn_model(self.algorithm)
                else:
                    raise Exception("sklearn generating must have a valid algorithm")
            elif self.model_type == "pytorch":
                model = self.gen_pytorch_model()
            else:
                model = self.gen_tensorflow_model()
        else:
            raise Exception(f"Not valid model implementation {self.model_type}")
        return model

    def gen_optimizer(self):
        if self.optimizer:
            if self.value_dict['optimizer']:
                return self.value_dict['optimizer']
            if self.valid:
                if self.model_type != 'pytorch':
                    raise Exception(f"Only pytorch models use optimizers")
                model = self.gen_model()
                return SGD([{'params': model.parameters()}, ], lr=0.001, momentum=0.85)
            else:
                class MYoptim:
                    def __init__(self):
                        pass

                return MYoptim
        else:
            return None

    def gen_loss(self):
        if self.loss:
            if self.value_dict['loss']:
                return self.value_dict['loss']
            if self.valid:
                if self.model_type == 'tensorflow':
                    return Loss()
                elif self.model_type == 'pytorch':
                    return _Loss()
                else:
                    raise Exception(f'Expected tensorflow or pytorch, got {self.model_type}')
            else:
                class MYloss:
                    def __init__(self):
                        pass

                return MYloss()

        else:
            return None

    def gen_input_val_range(self):
        if self.input_val_range:
            if self.value_dict['input_val_range']:
                return self.value_dict['input_val_range']
            if self.valid:
                return (1, 2.5)
            else:
                return (1, 2, 5)
        else:
            return None

    def gen_input_shape(self):
        if self.input_shape:
            if self.value_dict['input_shape']:
                return self.value_dict['input_shape']
            if self.valid:
                return (10, 30)
            else:
                return (1, 2.44)
        else:
            return None

    def gen_num_classes(self):
        if self.nb_classes:
            if self.value_dict['nb_classes']:
                return self.value_dict['nb_classes']
            if self.valid:
                return 10
            else:
                return 2.44
        else:
            return None

    def gen_dataloader(self):
        if self.value_dict['dataloader']:
            return self.value_dict['dataloader']
        if self.dataloader_type in self.valid_dataloader_type:
            if self.dataloader_type == 'list':
                return []
            elif self.dataloader_type == 'DataFrame':
                return pd.DataFrame()
            elif self.dataloader_type == 'array':
                return np.array([1, 2, 3])
            elif self.dataloader_type == 'ndarray':
                return np.ndarray([1, 2, 3])
            else:
                return Dataset()
        else:
            return None

    def gen_test_params(self):
        param_dict = {}
        param_dict['model'] = self.gen_model()
        param_dict['dataloader'] = self.gen_dataloader()
        param_dict['loss'] = self.gen_loss()
        param_dict['optimizer'] = self.gen_optimizer()
        param_dict['input_shape'] = self.gen_input_shape()
        param_dict['input_val_range'] = self.gen_input_val_range()
        param_dict['num_of_classes'] = self.gen_num_classes()
        return param_dict

    def gen_metadata(self):
        metadata = {'ML_model': {
                      'meta': {'file_id': 'ML_model', 'model_type': self.model_type,
                               'ML_type': self.ml_type, 'algorithm': self.algorithm},
                      'input': {'input_shape': self.gen_input_shape(),
                                'input_val_range': self.gen_input_val_range(),
                                'num_of_classes': self.gen_num_classes()},
                      'loss': {'meta': {'file_id': 'loss', 'loss_type': self.loss}},
                      'optimizer': {'meta': {'file_id': 'optimizer', 'optimizer_type': self.optimizer}}
                      },
                      'dataloader': {
                              'meta': {'file_id': 'dataloader', 'data_loader_type': self.dataloader_type}
                              },
                      'requirements.txt': {'meta': {'file_id': 'requirements.txt'
                        }
                      },
                      'authentication': {
                                      'bucket_name': 'mabdata207125196', 'access_key_id': 'key1',
                                      'secret_access_key': 'Skey1', 'region': 'us'
                              }
                    }

        json_metadada = json.dumps(metadata)
        return json_metadada

    def run_test(self):
        metadata = self.gen_metadata()
        model = self.gen_model()
        loss = self.gen_loss()
        optimizer = self.gen_optimizer()
        dataloader = self.gen_dataloader()
        loader = Bucket_loader()
        loader.upload(obj=model, obj_type="ML_model")
        loader.upload(obj=loss, obj_type='loss')
        loader.upload(obj=optimizer, obj_type='optimizer')
        loader.upload(obj=dataloader, obj_type='dataloader')
        with open('requirements.txt', 'r') as requirements:
            loader.upload(obj=requirements, obj_type='requirements.txt', to_pickle=False)
        # Downloading the requierments.txt file from S3 to colab
        loader.get_requirements()
        env_setter = Envsetter("requirements.txt")
        # Installing the file
        # env_setter.install_requirements()
        input_validatior = Input_validatior()
        if input_validatior.validate():
            input = input_validatior.get_input()
            wrapper = Estimator_handler(input)
            wrapper.wrap()
            print('Test passed!')
            return True
        else:
            print('Test faild! Invalid input from user!')
            return False


if __name__ == '__main__':
    value_dict = {'model': None, 'data_loader': None,
                  'loss': None, 'input_shape': None, 'nb_classes': None,
                  'input_val_range': False, 'algorithm': None, 'optimizer': False}
    test = Test(model_type='pytorch', dataloader_type='list', loss=True, input_shape=True, nb_classes=True)
    test.run_test()



