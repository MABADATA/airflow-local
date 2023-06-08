import logging
import json
import pickle
import os
from gcloud import storage
from user_files.model.model_helpers import get_model_package_root
from user_files.dataloader.dataloader_helpers import get_dataloader_package_root
from user_files.helpers import get_files_package_root

class BucketLoader:
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
    def __init__(self,metadata):
        """
        The bucket loader is an object that connect with aws and preform downloads
        and uploads from GCP bucket.

        Parameters
        ----------
        metadata : str/dict
            The json from user containing all metadata and structure as above.

            Expected json file:

            { ML_model:{
                      meta:{file_id: str(id),script_file_id: str(id),class_name: str(name), model_type: str(type), ML_type: str(type), algorithm: str(algorithm)},
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
                      "file_URL" : "URL to file"
                      "script_file_id": "ML_module",
                      "script_file_URL": "URL to file",
                      "class_name" : "class",
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
                        "URL": URL to file
                      }
                    },
                    "optimizer": {
                      "meta": {
                        "file_id": "None",
                        "optimizer_type": "None"
                        "URL": "URL to file"
                      }
                    }
                  },
                  "dataloader": {
                    "meta": {
                      "file_id": "Dataloader_file",
                      "file_URL": "URL to file"
                      "data_loader_type": "pandad.DataFrame"
                      "script_file_id": "Dataloader",
                      "script_file_URL" : "URL to file"
                      "class_name" : "class",

                    },
                    "dataset": {
                        "test_set_id" : "id",
                        "test_set_URL": "URL to file"
                    }
                  },
                  "requirements.txt": {
                    "meta": {
                      "file_id": "requirements.txt.txt"
                      "URL": URL to file
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

        self.metadata = metadata
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)
        if isinstance(self.metadata, dict):
            self.__ML_model_file_id = self.metadata['ML_model']['meta']['file_id']
            self.__ML_model_file_URL = self.metadata['ML_model']['meta']['file_URL']
            self.__ML_model_script_file_id = self.metadata['ML_model']['meta']['script_file_id']
            self.__ML_model_script_file_URL = self.metadata['ML_model']['meta']['script_file_URL']
            self.__ML_model_class_name = self.metadata['ML_model']['meta']['class_name']
            self.__loss_function_file_id = self.metadata['ML_model']['loss']['meta']['file_id']
            self.__loss_function_file_URL = self.metadata['ML_model']['loss']['meta']['URL']
            self.__optimizer_file_id = self.metadata['ML_model']['optimizer']['meta']['file_id']
            self.__optimizer_file_URL = self.metadata['ML_model']['optimizer']['meta']['URL']
            self.__dataloader_file_id = self.metadata['dataloader']['meta']['file_id']
            self.__dataloader_file_URL = self.metadata['dataloader']['meta']['file_URL']
            self.__dataloader_script_file_URL = self.metadata['dataloader']['meta']['script_file_URL']
            self.__requirements_file_id = self.metadata['requirements.txt']['meta']['file_id']
            self.__requirements_file_URL = self.metadata['requirements.txt']['meta']['URL']
            self.__account_service_key = self.metadata['authentication']['account_service_key']
            os.environ["BUCKET_NAME"] = self.metadata['authentication']['bucket_name']
            os.environ["ACOUNT_SERVICE_KEY"] = "account_service_key.json"
            with open("account_service_key.json", 'w') as f:
                json.dump(self.__account_service_key, f)
        else:
            raise TypeError('meta data need to be type dict or str')

    def __getstate__(self):
        state = {"__ML_model_file_id": self.__ML_model_file_id,
                 "__ML_model_script_file_id": self.__ML_model_script_file_id,
                 "__ML_model_class_name": self.__ML_model_class_name,
                 "__loss_function_file_id" : self.__loss_function_file_id,
                "__optimizer_file_id": self.__optimizer_file_id ,
                 "__dataloader_file_id" : self.__dataloader_file_id,
                 "__requirements_file_id": self.__requirements_file_id,
                 "__account_service_key": self.__account_service_key}
        return state

    def __setstate__(self, state):
        self.__ML_model_class_name = state['__ML_model_class_name']
        self.__ML_model_script_file_id = state['__ML_model_script_file_id']
        self.__ML_model_class_name = state['__ML_model_class_name']
        self.__loss_function_file_id = state['__loss_function_file_id']
        self.__optimizer_file_id = state['__optimizer_file_id']
        self.__dataloader_file_id = ['__dataloader_file_id']
        self.__requirements_file_id = state['__requirements_file_id']
        self.__account_service_key = state['__account_service_key']

    @staticmethod
    def get_client():
        ACOUNT_SERVICE_KEY = os.environ.get('ACCOUNT_SERVICE_KEY')
        BUCKET_NAME = os.environ.get('BUCKET_NAME')
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACOUNT_SERVICE_KEY
        os.environ["DONT_PICKLE"] = 'False'
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        return bucket

    @staticmethod
    def upload_to_gcp(dest_file_path, src_file_name):
        bucket = BucketLoader.get_client()
        blob = bucket.blob(dest_file_path)
        blob.upload_from_filename(src_file_name)

    @staticmethod
    def download_from_gcp(src_file_name,dest_file_name=None):
        bucket = BucketLoader.get_client()
        blob = bucket.blob(src_file_name)
        if dest_file_name:
            blob.download_to_filename(dest_file_name)
        blob.download_to_filename(src_file_name)


    # Getters starts here
    # All getters methods are downloading files from GCP bucket.

    def get_model(self):
        try:
            #download model source code file
            model_script_dest = get_model_package_root() + "/model_def.py"
            BucketLoader.download_from_gcp(src_file_name=self.__ML_model_script_file_URL,
                                           dest_file_name=model_script_dest)
            # downloading model pickle file
            model_bin_dest = get_model_package_root() + f"/{self.__ML_model_file_id}.pickle"
            BucketLoader.download_from_gcp(src_file_name=self.__ML_model_file_id,
                                           dest_file_name=model_bin_dest)
            file_name = f"{self.__ML_model_file_id}.pickle"
            model_dest = get_model_package_root() + file_name
            self.download_from_gcp(src_file_name=self.__ML_model_file_URL,
                                   dest_file_name=model_dest)

        except Exception as err:
            logging.error(f"Downloading ML model from GCP bucket failed!\nError:\n{err}")

    def get_file(self, file):
        """
        Function to get all files in the bucket ecxc
        :param file_name:
        :param file_format:
        :return:
        """
        file_name, file_format = file.split(".")
        if file_name + file_format == self.__loss_function_file_id:
            url = self.__loss_function_file_URL
        elif file_name + file_format == self.__optimizer_file_id:
            url = self.__optimizer_file_URL
        elif file_name + file_format == self.__requirements_file_id:
            url = self.__requirements_file_URL
        else:
            raise Exception(f"Expected loss, optimizer or requirements. Got{file_name}")
        try:
            loss_dest = get_files_package_root() + file_name + file_format
            self.download_from_gcp(src_file_name=url,
                                   dest_file_name=loss_dest)
        except Exception as err:
            logging.error(f"Downloading {file_name} from GCP bucket failed!\nError:\n{err}")

    def get_dataloader(self):
        try:
            #downloading dataloader source code file
            dataloader_script_dest = get_dataloader_package_root() + "/dataloader_def.py"
            self.download_from_gcp(src_file_name=self.__dataloader_script_file_URL,
                                   dest_file_name=dataloader_script_dest)

            # downloading dataloader file
            dataloader_dest = get_dataloader_package_root() + f"{self.__dataloader_file_id}.pickle"
            self.download_from_gcp(src_file_name=self.__dataloader_file_URL,
                                   dest_file_name=dataloader_dest)
        except Exception as err:
            logging.error(f"Downloading dataloader from GCP bucket failed!\nError:\n{err}")





    # def upload(self, obj, obj_type, to_pickle=True):
    #     hashmap = {"ML_model": self.__ML_model_file_id,
    #                "loss": self.__loss_function_file_id,
    #                "optimizer": self.__optimizer_file_id,
    #                "dataloader": self.__dataloader_file_id,
    #                "requirements.txt": self.__requirements_file_id,
    #                "Estimator_params": "Estimator_params.json",
    #                "attack_defence_metadata": "attack_defence_metadata.json"}
    #     try:
    #         file_name = hashmap[obj_type]
    #         if to_pickle:
    #             logging.info(f'Dumping {obj_type} to pickle file...')
    #             self.to_pickle(file_name, obj)
    #             file_name += ".pickle"
    #         logging.info(f'Uploading {obj_type} to GCP bucket starts...')
    #         self.upload_to_gcp(file_name, file_name)
    #         logging.info('Upload was successful!')
    #     except Exception as err:
    #         logging.error(f"Uploading {obj_type} to GCP failed!\nError:\n{err}")

