from Bucket_loader import Bucket_loader
import json
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression.pytorch import PyTorchRegressor
from art.estimators.classification import TensorFlowClassifier, TensorFlowV2Classifier
from art.estimators.regression.keras import KerasRegressor
from art.estimators.regression.scikitlearn import ScikitlearnDecisionTreeRegressor
from art.estimators.classification.scikitlearn import (ScikitlearnAdaBoostClassifier,
  ScikitlearnBaggingClassifier,
  ScikitlearnDecisionTreeClassifier,
  ScikitlearnExtraTreesClassifier,
  ScikitlearnGradientBoostingClassifier,
  ScikitlearnLogisticRegression,
  ScikitlearnRandomForestClassifier,
  ScikitlearnSVC,
  ScikitlearnGaussianNB)

class Estimator_handler:
    def __init__(self, input, json_meta_data):
        if isinstance(json_meta_data, str):
            json_meta_data = json.loads(json_meta_data)
        if isinstance(json_meta_data, dict):
            self.metadata = json_meta_data
        else:
            raise TypeError('meta data need to be type dict or str')
        self.__file_loader = Bucket_loader(self.metadata)
        self.__ml_type = input["ML_type"]
        self.__implementation = input["implementation"]
        self.__algorithm = input['algorithm']
        self.__loss = input['Loss']
        self.__optimizer = input['Optimizer']
        self.__input_shape = self.metadata['ML_model']['input']['input_shape']
        self.__input_val_range = self.metadata['ML_model']['input']["input_val_range"]
        self.__num_of_classes = self.metadata['ML_model']['input']["num_of_classes"]
        self.estimator = None  # get a value after self._wrap_model will run
        self.map = {"implementation": {
            "pytorch": {
                "ML_type": {
                    "classification": PyTorchClassifier,
                    "regression": PyTorchRegressor
                }
            },
            "tensorflow": {
                "ML_type": {
                    "classification": TensorFlowV2Classifier,
                    "regression": KerasRegressor

                }
            },
            "sklearn": {
                "ML_type": {
                    "classification": {
                        "DecisionTree": ScikitlearnDecisionTreeClassifier,
                        "ExtraTree": ScikitlearnExtraTreesClassifier,
                        "AdaBoost": ScikitlearnAdaBoostClassifier,
                        "Bagging": ScikitlearnBaggingClassifier,
                        "GradientBoosting": ScikitlearnGradientBoostingClassifier,
                        "RandomForest": ScikitlearnRandomForestClassifier,
                        "LogisticRegression": ScikitlearnLogisticRegression,
                        "GaussianNB": ScikitlearnGaussianNB,
                        "SVC": ScikitlearnSVC,
                        "LinearSVC": ScikitlearnSVC,
                    },
                    "regression": {
                        "DecisionTree":  ScikitlearnDecisionTreeRegressor}}
            }
        }

        }

    def _estimator_match(self):
        if self.__implementation == 'sklearn':
            estimator = self.map['implementation']['sklearn']['ML_type'][self.__ml_type][self.__algorithm]
        else:
            estimator = self.map['implementation'][self.__implementation]['ML_type'][self.__ml_type]
        return estimator

    def _wrap_model(self, estimator_obj):
        my_model = self.__file_loader.get_model()
        if self.__implementation == 'sklearn':
            my_estimator = estimator_obj(model=my_model, clip_values=self.__input_val_range)
        elif self.__implementation == 'tensorflow':
            if self.__ml_type == "regression":
                my_estimator = estimator_obj(model=my_model)
            elif self.__ml_type == "classification":
                my_estimator = estimator_obj(model=my_model,
                                             nb_classes=self.__num_of_classes,
                                             input_shape=self.__input_shape)
            else:
                raise Exception("Can't wrap model!\nML type must be classification or regression")
        elif self.__implementation == 'pytorch':
            optimizer = None
            if self.metadata['ML_model']['optimizer']['meta']['optimizer_type']:
                optimizer = self.__file_loader.get_optimizer()
            loss = self.__file_loader.get_loss()
            if self.__ml_type == "regression":
                my_estimator = estimator_obj(model=my_model, loss=loss,
                                             input_shape=self.__input_shape,
                                             optimizer=optimizer)
            elif self.__ml_type == "classification":
                my_estimator = estimator_obj(model=my_model, loss=loss,
                                             nb_classes=self.__num_of_classes,
                                             input_shape=self.__input_shape,
                                             optimizer=optimizer)
            else:
                raise Exception("Can't wrap model!\nML type must be classification or regression")

        else:
            raise Exception("Can't wrap model!\nImplementation type must be sklearn,pytorch or tensorflow")

        return my_estimator

    def wrap(self):
        estimator_obj = self._estimator_match()
        wrap_model = self._wrap_model(estimator_obj)
        self.__file_loader.upload(wrap_model, "ML model")
