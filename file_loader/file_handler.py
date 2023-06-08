import logging
import json
import torch.nn as nn
import pickle
import os
from user_files.dataloader.dataloader_def import *
from file_loader.bucket_loader import BucketLoader
import Preprocess.Input_validation
from user_files.model.model_helpers import get_model_package_root
from user_files.dataloader.dataloader_helpers import get_dataloader_package_root
from user_files.helpers import get_files_package_root
from torch.optim import SGD
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression.pytorch import PyTorchRegressor
from art.estimators.classification import TensorFlowV2Classifier
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
class FileLoader:

    def __init__(self,metadata):
        self.bucket_loader = BucketLoader(metadata)
        self.from_bucket = os.environ.get('FROM_BUCKET')
        self.metadata = metadata
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)
        if isinstance(self.metadata, dict):
            self.__ML_model_file_id = self.metadata['ML_model']['meta']['file_id']
            self.__loss_function_file_id = self.metadata['ML_model']['loss']['meta']['file_id']
            self.__optimizer_file_id = self.metadata['ML_model']['optimizer']['meta']['file_id']
            self.__dataloader_file_id = self.metadata['dataloader']['meta']['file_id']
            self.__requirements_file_id = self.metadata['requirements.txt']['meta']['file_id']
            self.input_validator = Preprocess.Input_validation.InputValidator(metadata)
    @staticmethod
    def to_pickle(file_name,obj):
        with open(f'./{file_name}.pickle', 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def from_pickle(file_name):
        with open(file_name, 'rb') as f:
            loaded_obj = pickle.load(f)
            return loaded_obj
    def check_file_in_local(self,dir_path, expected_file):
        scaner = os.scandir(path=dir_path)
        for entry in scaner:
            if entry.is_dir() or entry.is_file():
                if entry.name == expected_file:
                    return True
        return False

    def get_model(self):
        def get_model_from_local(expected_file):
            bin_model_path = get_model_package_root() + expected_file
            model = FileLoader.from_pickle(bin_model_path)
            if self.input_validator.validate_model(model):
                return model
            raise Exception("The model is not valid")

        # check if the ML model's file is in the folder
        expected_file = f"/{self.__ML_model_file_id}.pickle"
        file_in_local = self.check_file_in_local(dir_path=get_model_package_root(),expected_file=expected_file)

        if file_in_local:
            try:
                model = get_model_from_local(expected_file)
                return model
            except Exception as err:
                logging.error(f"Error occurred while getting ML model from local:\nError: {err}")
                return
        elif self.from_bucket:
            try:
                self.bucket_loader.get_model()
            except Exception as err:
                raise Exception(f"Failed to get ML model from bucket:\nError: {err}")

            try:
                model = get_model_from_local(expected_file)
                return model
            except Exception as err:
                logging.error(f"Error occurred while getting ML model from local:\nError: {err}")
                return
        else:
            raise FileNotFoundError("ML model file not found")

    def get_dataloader(self):
        def get_dataloader_from_local(expected_file):
            bin_dataloader_path = get_dataloader_package_root() + expected_file
            dataloader = FileLoader.from_pickle(bin_dataloader_path)
            if self.input_validator.validate_dataloder(dataloader):
                return dataloader
            raise Exception("The dataloader is not valid")

        # check if the ML model's file is in the folder
        expected_file = f"/{self.__dataloader_file_id}.pickle"
        file_in_local = self.check_file_in_local(dir_path=get_dataloader_package_root(),expected_file=expected_file)

        if file_in_local:
            try:
                dataloader = get_dataloader_from_local(expected_file)
                return dataloader
            except Exception as err:
                logging.error(f"Error occurred while getting dataloader from local:\nError: {err}")
                return
        elif self.from_bucket:
            try:
                self.bucket_loader.get_dataloader()
            except Exception as err:
                raise Exception(f"Failed to get dataloader from bucket:\nError: {err}")

            try:
                dataloader = get_dataloader_from_local(expected_file)
                return dataloader
            except Exception as err:
                logging.error(f"Error occurred while getting dataloader from local:\nError: {err}")
                return
        else:
            raise FileNotFoundError("dataloader file not found")


    def get_file(self, expected_file):
        def get_file_from_local(expected_file):
            file_path = get_files_package_root() + expected_file
            if isinstance(expected_file, str):
                file_name, file_format = expected_file.split(".")
                if file_format == "pickle" or file_format == "pkl":
                    file = FileLoader.from_pickle(file_path)
                    return file
                elif file_format == 'json':
                    with open(file_path, 'r') as f:
                        return json.load(f)
                elif file_format == 'txt':
                    with open(file_path, 'r') as f:
                        return f.read()
                else:
                    raise Exception(f"Expected file format pickle,pkl,json or txt. got {file_format}")

        # check if the ML model's file is in the folder
        file_in_local = self.check_file_in_local(dir_path=get_files_package_root(),expected_file=expected_file)

        if file_in_local:
            try:
                file = get_file_from_local(expected_file)
                return file
            except Exception as err:
                logging.error(f"Error occurred while getting {expected_file} from local:\nError: {err}")
                return
        elif self.from_bucket:
            try:
                self.bucket_loader.get_file(expected_file)
                file =  get_file_from_local(expected_file)
                return file
            except Exception as err:
                raise Exception(f"Failed to get dataloader from bucket:\nError: {err}")
        else:
            raise FileNotFoundError(f""
                                    f"{expected_file} file not found")


    def get_estimator(self):
        def get_obj_from_str(obj_as_str):
            estimator_map = {
                "PyTorchClassifier": PyTorchClassifier,
                "PyTorchRegressor": PyTorchRegressor,
                "TensorFlowV2Classifier": TensorFlowV2Classifier,
                "KerasRegressor": KerasRegressor,
                "ScikitlearnDecisionTreeClassifier": ScikitlearnDecisionTreeClassifier,
                "ScikitlearnExtraTreesClassifier": ScikitlearnExtraTreesClassifier,
                "ScikitlearnAdaBoostClassifier": ScikitlearnAdaBoostClassifier,
                "ScikitlearnBaggingClassifier": ScikitlearnBaggingClassifier,
                "ScikitlearnGradientBoostingClassifier": ScikitlearnGradientBoostingClassifier,
                "ScikitlearnRandomForestClassifier": ScikitlearnRandomForestClassifier,
                "ScikitlearnLogisticRegression": ScikitlearnLogisticRegression,
                "ScikitlearnGaussianNB": ScikitlearnGaussianNB,
                "ScikitlearnSVC": ScikitlearnSVC,
                "ScikitlearnDecisionTreeRegressor": ScikitlearnDecisionTreeRegressor
            }
            return estimator_map[obj_as_str]

        def assign_vars(cls, args_dict, ML_model):
            if args_dict.get("optimizer"):
                optimizer = SGD(ML_model.parameters(), lr=0.01)
                args_dict["optimizer"] = optimizer
            if args_dict.get("loss"):
                try:
                    loss = self.get_file(self.__loss_function_file_id)
                    # only from test
                    args_dict["loss"] = loss
                except:
                    args_dict["loss"] = nn.CrossEntropyLoss()
            obj = cls(**args_dict, model=ML_model)
            return obj
        logging.info("Getting estimator...")
        with open(get_files_package_root() + "/Estimator_params.json", 'r') as f:
            estimator_params = json.load(f)
        model = self.get_model()
        # model = NeuralNetworkClassificationModel(29,2)
        estimator_str = estimator_params['object']
        estimator_obj = get_obj_from_str(estimator_str)
        params = estimator_params['params']
        estimator = assign_vars(cls=estimator_obj, args_dict=params, ML_model=model)
        return estimator

    def save_file(self, obj, path, as_pickle=False, as_json=False):
        """
        saves the file
        :param path: path to save the file in
        :param bin: if the file is binary or not
        :return:
        """
        if as_pickle:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        elif as_json:
            with open(path, 'w') as f:
                json.dump(obj, f)

        else:
            raise Exception("Expected as_pickle or as_json to be True")


# if __name__ == '__main__':
#     d = {
#         "ML_model": {
#             "meta": {
#                 "file_id": "ML_file",
#                 "script_file_id": "ML_module",
#                 "class_name" : "class",
#                 "model_type": "sklearn",
#                 "ML_type": "classification",
#                 "algorithm": "randomForest",
#                 "URL" : "path to file"
#             },
#             "input": {
#                 "input_shape": "(100,20)",
#                 "input_val_range": "(10,20)",
#                 "num_of_classes": None
#             },
#             "loss": {
#                 "meta": {
#                     "file_id": None,
#                     "loss_type": None,
#                     "URL": "path to file"
#                 }
#             },
#             "optimizer": {
#                 "meta": {
#                     "file_id": None,
#                     "optimizer_type": None,
#                     "URL": "path to file"
#                 }
#             }
#         },
#         "dataloader": {
#             "meta": {
#                 "file_id": "Dataloader_file",
#                 "data_loader_type": 'pandas.DataFrame',
#                 "URL": "path to file"
#             }
#         },
#         "requirements.txt": {
#             "meta": {
#                 "file_id": "requirements.txt",
#                 "URL": "path to file"
#             }
#
#         },
#         "authentication": {
#             "bucket_name": "MyBucket",
#             "account_service_key": {}
#         }
#     }
