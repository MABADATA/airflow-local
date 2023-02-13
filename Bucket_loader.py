import logging
import json
import pickle
import os
from gcloud import storage

ACOUNT_SERVICE_KEY=r"C:\Users\Administrator\Desktop\mabadata-733abc189d01.json"
BUCKET_NAME='mabdata207125196'


if __name__ == '__main__':
    class test:
        def __init__(self):
            self.val = 4
    def to_pickle(file_name,obj):
        with open(f'./{file_name}.pickle', 'wb') as f:
            pickle.dump(obj, f)
    t = test()
    to_pickle('mytest',t)
class Bucket_loader:
    def __init__(self, meta_data):
        """
        The bucket loader is an object that connect with aws and preform downloads
        and uploads from S3 bucket.

        Parameters
        ----------
        meta_data : str\dict
            The json from user containig all meta data and structure as above.

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
                  "access_key_id": "2k4k2jU992",
                  "secret_access_key": "882LPh87",
                  "region": "us.central"
                  }
                }



        Methods
        ----------
        All the methods are getter, setters and one method of that open connection.
        Getters dwonloading files from S3 bucket, while setters uploading files to S3.



        """
        if isinstance(meta_data, str):
            meta_data = json.loads(meta_data)
        if isinstance(meta_data, dict):
            self.__bucket = meta_data['authentication']['bucket_name']
            self.__access_key_id = meta_data['authentication']['access_key_id']
            self.__secret_access_key = meta_data['authentication']['secret_access_key']
            self.__region = meta_data['authentication']['region']
            self.__ML_model_file_id = meta_data['ML_model']['meta']['file_id']
            self.__loss_function_file_id = meta_data['ML_model']['loss']['meta']['file_id']
            self.__optimizer_file_id = meta_data['ML_model']['optimizer']['meta']['file_id']
            self.__dataloader_file_id = meta_data['dataloader']['meta']['file_id']
            self.__requirements_file_id = meta_data['requirements.txt']['meta']['file_id']
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
    def download_from_gcp(src_file_name):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACOUNT_SERVICE_KEY
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(src_file_name)
        blob.download_to_filename(src_file_name)
    @staticmethod
    def to_pickle(file_name,obj):
        with open(f'./{file_name}.pickle', 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def from_pickle(file_name):
        with open(f'{file_name}.pickle','rb') as f:
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
            file_name = "requirements.txt"
            self.download_from_gcp(file_name)
            with open(file_name, 'r') as f:
                return f.read()
        except Exception as err:
            logging.error(f"Downloading requirements.txt file from GCP bucket failed!\nError:\n{err}")

    # Setters starts here
    def upload(self, obj, obj_type, to_pickle=True):
        hashmap = {"ML model": self.__ML_model_file_id,
                   "loss": self.__loss_function_file_id,
                   "optimizer": self.__optimizer_file_id,
                   "dataloader": self.__dataloader_file_id,
                   "requirements.txt": self.__requirements_file_id}
        try:
            file_name = hashmap[obj_type]
            if to_pickle:
                logging.info(f'Dumping {obj_type} to pickle file...')
                self.to_pickle(file_name,obj)
                file_name += ".pickle"
            logging.info(f'Uploading {obj_type} to GCP bucket starts...')
            self.upload_to_gcp(file_name, file_name)
            logging.info('Upload was successful!')
        except Exception as err:
            logging.error(f"Uploading {obj_type} to S3 failed!\nError:\n{err}")


