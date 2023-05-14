import logging
import json
import pickle
import os
import torch.nn as nn
from gcloud import storage
from torch.optim import Optimizer,SGD
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
ACOUNT_SERVICE_KEY=r""
BUCKET_NAME=''


if __name__ == '__main__':
    class test:
        def __init__(self):
            self.val = 4
    def to_pickle(file_name,obj):
        with open(f'./{file_name}.pickle', 'wb') as f:
            pickle.dump(obj, f)
    # with open("Estimator.pickle", 'rb') as f:
    #     loaded_obj = pickle.load(f)
    #     print(loaded_obj)
    # t = test()
    # to_pickle('mytest',t)
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACOUNT_SERVICE_KEY
    # storage_client = storage.Client()
    # new_bucket = storage_client.create_bucket('user2071251796')
    # # bucket = storage_client.bucket(BUCKET_NAME)
    # blob = new_bucket.blob('mytest.pickle')
    # blob.upload_from_filename("mytest.pickle")


class Bucket_loader:
    """
    Attributes:
    __________
    bucket: str, The name of the bucket holding the files
    access_key_id: str, access key to the GCP account
    secret_access_key: str, secret access key to the GCP account
    region: Bucket region
    ML_model_file_id: str, name of the file containing the ML model without the file type(e.g pickle)
    loss_function_file_id: str, name of the file containing the loss function without the file type(e.g pickle)
    optimizer_file_id: str, name of the file containing the optimizer without the file type(e.g pickle)
    dataloader_file_id: str, name of the file containing the dataloader without the file type(e.g pickle)
    requirements_file_id: str, name of the file containing the requirements with the file type(e.g txt)

    Methods:
    --------
    upload_to_gcp(): static method, uploading files to GCP bucket.
    download_from_GCP(): static method, downloading files from GCP bucket.
    to_pickle(): static method, converting objects to pickle files.
    from_pickle(): static method, converting pickle files to objects.
    upload(): wrapper to upload_to_gcp method, that handles different.
    get_model(): returning the ML model from GCP.
    get_loss(): returning the loss function as an object from GCP.
    get_optimizer(): returning the optimizer as an object from GCP.
    get_dataloader(): returning the dataloader as an object from GCP.
    get_requirements(): returning the requirements file as txt from GCP.
    """
    def __init__(self):
        """
        The bucket loader is an object that connect with aws and preform downloads
        and uploads from GCP bucket.

        Parameters
        ----------
        meta_data : str/dict
            The json from user containing all metadata and structure as above.

            Expected json file:

            { ML_model:{
                      meta:{file_id: str(id), model_type: str(type), ML_type: str(type),algorithm: str(algorithm)},
                      input:{input_shape: tuple(rows,cols),input_val_range: tuple(min,max),num_of_classes: int(num)},
                      loss:{meta: {file_id: str(id), loss_type: str(type)}},
                      optimizer:{meta: {file_id: str(id), optimizer_type: str(type)}}
                      },
              dataloader:{
                      meta:{file_id: str(id), data_loader_type: str(type)}
                      },
              requirements.txt: {meta:{file_id: str(id)
                }
              },

              authentication:{
                              bucket_name: str(name), access_key_id: str(id),
                              secret_access_key: str(key),region: str(region)
                      }
            }

          Example:
                {
                  "ML_model": {
                    "meta": {
                      "file_id": "ML_file",
                      "model_type": "sklearn",
                      "ML_type": "classification",
                      "algorithm": "randomForest"
                    },
                    "input": {
                      "input_shape": "(100,20)",
                      "input_val_range": "(10,20)"
                      "num_of_classes": None
                    },
                    "loss": {
                      "meta": {
                        "file_id": "None",
                        "loss_type": "None"
                      }
                    },
                    "optimizer": {
                      "meta": {
                        "file_id": "None",
                        "optimizer_type": "None"
                      }
                    }
                  },
                  "dataloader": {
                    "meta": {
                      "file_id": "Dataloader_file",
                      "data_loader_type": "pandad.DataFrame"
                    }
                  },
                  "requirements.txt": {
                    "meta": {
                      "file_id": "requirements.txt.txt"
                    }

                  },
                  "authentication": {
                  "bucket_name": "MyBucket",
                  "account_service_key": "ACOUNT_SERVICE_KEY.json"
                  }
                }



        Methods
        ----------
        All the methods are getter, setters and one method of that open connection.
        Getters dwonloading files from S3 bucket, while setters uploading files to GCP.



        """
        def get_meta_data():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACOUNT_SERVICE_KEY
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob('meta_data.json')
            blob.download_to_filename('meta_data.json')
            metadata = json.loads('meta_data.json')
            return metadata

        self.metadata = get_meta_data()
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)
        if isinstance(self.metadata, dict):
            self.__ML_model_file_id = self.metadata['ML_model']['meta']['file_id']
            self.__loss_function_file_id = self.metadata['ML_model']['loss']['meta']['file_id']
            self.__optimizer_file_id = self.metadata['ML_model']['optimizer']['meta']['file_id']
            self.__dataloader_file_id = self.metadata['dataloader']['meta']['file_id']
            self.__requirements_file_id = self.metadata['requirements.txt']['meta']['file_id']
            self.__account_service_key = self.metadata['authentication']['account_service_key']
            BUCKET_NAME = self.metadata['authentication']['bucket_name']
            ACOUNT_SERVICE_KEY = "account_service_key.json"
            with open(ACOUNT_SERVICE_KEY, 'w') as f:
                json.dump(self.__account_service_key, f)
        else:
            raise TypeError('meta data need to be type dict or str')

    @staticmethod
    def upload_to_gcp(dest_file_path, src_file_name):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACOUNT_SERVICE_KEY
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(dest_file_path)
        blob.upload_from_filename(src_file_name)

    @staticmethod
    def download_from_gcp(src_file_name,as_json=False):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACOUNT_SERVICE_KEY
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(src_file_name)
        if as_json:
            json_string = blob.download_as_string()
            return json_string
        blob.download_to_filename(src_file_name)
    @staticmethod
    def to_pickle(file_name,obj):
        with open(f'./{file_name}.pickle', 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def from_pickle(file_name):
        with open(file_name, 'rb') as f:
            loaded_obj = pickle.load(f)
            return loaded_obj


    # Getters starts here
    # All getters methods are downloading files from GCP bucket.

    def get_model(self):
        try:
            file_name = f"{self.__ML_model_file_id}.pickle"
            self.download_from_gcp(file_name)
            model = self.from_pickle(file_name)
            return model
        except Exception as err:
            logging.error(f"Downloading ML model from GCP bucket failed!\nError:\n{err}")

    def get_loss(self):
        try:
            if self.__loss_function_file_id:
                file_name = f"{self.__loss_function_file_id}.pickle"
                self.download_from_gcp(file_name)
                loss = self.from_pickle(file_name)
                return loss
            logging.info('Loss function not provided')
            return
        except Exception as err:
            logging.error(f"Downloading loss function from GCP bucket failed!\nError:\n{err}")

    def get_optimizer(self):
        try:
            if self.__optimizer_file_id:
                file_name = f"{self.__optimizer_file_id}.pickle"
                self.download_from_gcp(file_name)
                optimizer = self.from_pickle(file_name)
                return optimizer
            logging.info('optimizer not provided')
            return
        except Exception as err:
            logging.error(f"Downloading optimizer from GCP bucket failed!\nError:\n{err}")

    def get_dataloader(self):
        try:
            file_name = f"{self.__dataloader_file_id}.pickle"
            self.download_from_gcp(file_name)
            dataloader = self.from_pickle(file_name)
            return dataloader
        except Exception as err:
            logging.error(f"Downloading dataloader from GCP bucket failed!\nError:\n{err}")

    def get_requirements(self):
        try:
            file_name = self.__requirements_file_id
            self.download_from_gcp(file_name)
            with open(file_name, 'r') as f:
                return f.read()
        except Exception as err:
            logging.error(f"Downloading requirements.txt file from GCP bucket failed!\nError:\n{err}")
    # Setters starts here
    def get_attack_defence_json(self):
        try:
            file_name = 'attack_defence_metadata.json'
            json_string = self.download_from_gcp(file_name,as_json=True)
            data = json.loads(json_string)
            return data
        except Exception as err:
            logging.error(f"Downloading attack_defence_metadata.json file from GCP bucket failed!\nError:\n{err}")

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
                    loss = self.get_loss()
                    # only from test
                    args_dict["loss"] = loss
                except:
                    args_dict["loss"] = nn.CrossEntropyLoss()
            obj = cls(**args_dict, model=ML_model)
            return obj
        logging.info("Getting estimator...")
        json_string = self.download_from_gcp("Estimator_params.json")
        estimator_params = json.loads(json_string)
        model = self.get_model()
        # model = NeuralNetworkClassificationModel(29,2)
        estimator_str = estimator_params['object']
        estimator_obj = get_obj_from_str(estimator_str)
        params = estimator_params['params']
        estimator = assign_vars(cls=estimator_obj, args_dict=params, ML_model=model)
        return estimator

    def upload(self, obj, obj_type, to_pickle=True):
        hashmap = {"ML_model": self.__ML_model_file_id,
                   "loss": self.__loss_function_file_id,
                   "optimizer": self.__optimizer_file_id,
                   "dataloader": self.__dataloader_file_id,
                   "requirements.txt": self.__requirements_file_id,
                   "Estimator_params": "Estimator_params.json",
                   "attack_defence_metadata": "attack_defence_metadata.json"}
        try:
            file_name = hashmap[obj_type]
            if to_pickle:
                logging.info(f'Dumping {obj_type} to pickle file...')
                self.to_pickle(file_name, obj)
                file_name += ".pickle"
            logging.info(f'Uploading {obj_type} to GCP bucket starts...')
            self.upload_to_gcp(file_name, file_name)
            logging.info('Upload was successful!')
        except Exception as err:
            logging.error(f"Uploading {obj_type} to GCP failed!\nError:\n{err}")


