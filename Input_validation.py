from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostClassifier,BaggingClassifier,
                              GradientBoostingClassifier,RandomForestClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer,SGD
import logging
import json
import torch.nn as nn
import collections
from Bucket_loader import Bucket_loader
import tensorflow as tf
import keras
from keras.losses import Loss

class Input_validatior:
    def __init__(self) -> None:
        self.__file_loader = Bucket_loader()
        if isinstance(self.__file_loader.metadata, str):
            self.metadata = json.loads(self.__file_loader.metadata)
        if isinstance(self.__file_loader.metadata, dict):
            self.metadata = self.__file_loader.metadata
        else:
            raise TypeError('meta data need to be type dict or str')
        self.__input = {"ML_type": self.metadata['ML_model']['meta']['ML_type']
            , "implementation": None, "algorithm": None,
                        "Loss": None, "Optimizer": None}

        self.map = {"ML_model":
            {"implementation":
                {
                    "pytorch": nn.Module,
                    "tensorflow": tf.keras.models.Model,
                    "sklearn": {
                        "class": BaseEstimator,
                        "algorithm": {
                            "classification": {
                                "DecisionTree": DecisionTreeClassifier,
                                "ExtraTree": ExtraTreeClassifier,
                                "AdaBoost": AdaBoostClassifier,
                                "Bagging": BaggingClassifier,
                                "GradientBoosting": GradientBoostingClassifier,
                                "RandomForest": RandomForestClassifier,
                                "LogisticRegression": LogisticRegression,
                                "GaussianNB": GaussianNB,
                                "SVC": SVC,
                                "LinearSVC": LinearSVC,
                            },
                            "regression": {
                                "DecisionTree": DecisionTreeRegressor}}
                    }
                }
            },
            "Loss": {
                "implementation": {
                    "pytorch": _Loss,
                    "tensorflow": Loss,
                }
            },
            "Optimizer": {
                "implementation": {
                    "pytorch": Optimizer,
                }
            }
        }

    def _validate_model(self):
        """
        validate the ML model input
        """
        try:
            model = self.__file_loader.get_model()
            # e.g pytorch, tensorflow
            model_implementation_type = self.metadata['ML_model']['meta']['model_type']
            self.__input['implementation'] = model_implementation_type
            if model_implementation_type == 'sklearn':
                if self.metadata['ML_model']['meta']['algorithm'] is not None:
                    # e.g regression, classification
                    input_ml_type = self.metadata['ML_model']['meta']['ML_type']
                    # e.g ExtraTree, SVC as string
                    input_algorithm = self.metadata['ML_model']['meta']['algorithm']
                    self.__input['algorithm'] = input_algorithm
                    # the class all sklearn models need to inhiret from(baseEstimaitor)
                    valid_ml_type = self.map['ML_model']['implementation']['sklearn']['class']
                    # the class of the algorithm the user uses
                    valid_algorithm_type = \
                    self.map['ML_model']['implementation']['sklearn']['algorithm'][input_ml_type][input_algorithm]
                    # returning if the model is on type BaseEstimator and the algorithm type
                    return isinstance(model, valid_ml_type) and isinstance(model, valid_algorithm_type)
                else:
                    return isinstance(model, self.map['sklearn']['class'])
            # model implementation is pytroch or tensorflow or other
            else:
                return isinstance(model, self.map['ML_model']['implementation'][model_implementation_type])
        except Exception as err:
            logging.error(f'Did not mannage to validate ML model, Error occurred\nError:\n{err}')
            return False

    def _validate_dataloder(self):
        try:
            dataloader = self.__file_loader.get_dataloader()
            return isinstance(dataloader, collections.abc.Iterable)
        except Exception as err:
            logging.error(f'Did not mannage to validate dataloder, Error occurred\nError:\n{err}')
            return False

    def _validate_loss_func(self):
        try:
            loss_func = self.__file_loader.get_loss()
            if loss_func:
                model_implementation_type = self.metadata['ML_model']['meta']['model_type']
                valid_loss_func_type = self.map['Loss']['implementation'][model_implementation_type]
                self.__input['Loss'] = valid_loss_func_type
                return isinstance(loss_func, self.map['Loss']['implementation'][model_implementation_type])
            else:
                logging.info("No loss function provided")
                # only pytorch has to have loss function object.
                if self.metadata['ML_model']['meta']['model_type'] == "pytorch":
                    return False
                return True
        except Exception as err:
            logging.error(f'Did not mannage to validate loss function, Error occurred\nError:\n{err}')
            return False

    def _validate_optimizer(self):
        try:
            optimizer = self.__file_loader.get_optimizer()
            if optimizer:
                model_implementation_type = self.metadata['ML_model']['meta']['model_type']
                valid_optimizer_type = self.map['Optimizer']['implementation'][model_implementation_type]
                self.__input['Optimizer'] = valid_optimizer_type
                return isinstance(optimizer, self.map['Optimizer']['implementation'][model_implementation_type])
            else:
                logging.info("No Optimizer provided")
                return True
        except Exception as err:
            logging.error(f'Did not mannage to validate optimzier, Error occurred\nError:\n{err}')
            return False

    def _validate_input_shape(self):
        try:
            shape = self.metadata['ML_model']['input']["input_shape"]
            is_shape = all([shape is not None,
                            (isinstance(shape, tuple) or isinstance(shape, list)) and
                            len(shape) == 2 and isinstance(shape[0], int) and
                            isinstance(shape[1], int)])
            return is_shape
        except Exception as err:
            logging.error(f'Did not mannage to validate parameter, Error occurred\nError:\n{err}')
            return False

    def _validate_num_of_classes(self):
        try:
            num_classes = self.metadata['ML_model']['input']["num_of_classes"]
            is_classes = num_classes is not None and isinstance(num_classes, int)
            return is_classes
        except Exception as err:
            logging.error(f'Did not mannage to validate parameter, Error occurred\nError:\n{err}')
            return False

    def _validate_range_of_vals(self):
        range = self.metadata['ML_model']['input']["input_val_range"]
        is_singel_range = all([range is not None,
                               isinstance(range, tuple) and
                               len(range) == 2 and isinstance(range[0], float) and
                               isinstance(range[1], float)], range[0] < range[1])
        try:
            is_var_of_ranges = [isinstance(r, tuple) and len(r) == 2 and
                                isinstance(r[0], float) and isinstance(r[1], float)
                                and r[0] < r[1] for r in range]
        except:
            is_var_of_ranges = [False]
        return is_singel_range or all(is_var_of_ranges)

    def _validatae_model_param(self):
        try:
            model_implementation_type = self.metadata['ML_model']['meta']['model_type']
            ml_type = self.metadata['ML_model']['meta']['ML_type']
            if model_implementation_type == "sklearn":
                # needed prarams are: clip_values --> range of values
                return self._validate_range_of_vals()
            elif model_implementation_type == "tensorflow":
                if ml_type == 'regression':
                    # need only the model
                    return True
                # needed params are: nb_classes --> number of classes as int,
                # input_shape --> tuple of ints
                return self._validate_num_of_classes() and self._validate_input_shape()
            elif model_implementation_type == "pytorch":
                if ml_type == 'regression':
                    # need input shape
                    return self._validate_input_shape()
                elif ml_type == "classification":
                    return self._validate_num_of_classes() and self._validate_input_shape()
                else:
                    logging.info(f"No such ML type {ml_type}")
                    return False
            else:
                logging.info(f"No such implementation: {model_implementation_type}")
        except Exception as err:
            logging.error(f'Did not manage to validate params, Error occurred\nError:\n{err}')
            return False

    def validate(self,print_res=True):
        model_validity = self._validate_model()
        loss_func_validity = self._validate_loss_func()
        optimizer_validity = self._validate_optimizer()
        dataloader_validity = self._validate_dataloder()
        model_params_validity = self._validatae_model_param()
        if print_res:
            print(f"""Validation results:\nmodel: {model_validity}\nloss function: {loss_func_validity}\noptimzer: {optimizer_validity}\ndataloader: {dataloader_validity}\nparams: {model_params_validity} """)
        return all([model_validity, loss_func_validity, optimizer_validity, dataloader_validity, model_params_validity])

    def get_input(self):
        if self.validate(print_res=False):
            return self.__input
        else:
            logging.info("Input is not valid!")
            return None

