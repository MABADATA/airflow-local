import logging
import json
import boto3
import pickle


# class Bucket_loader:
#     def __init__(self, meta_data):
#         """
#         The bucket loader is an object that connect with aws and preform downloads
#         and uploads from S3 bucket.
#
#         Parameters
#         ----------
#         meta_data : str\dict
#             The json from user containig all meta data and structure as above.
#
#             Expected json file:
#
#             { ML_model:{
#                       meta:{file_id: str(id), model_type: str(type), ML_type: str(type),algorithm: str(algorithm)},
#                       input:{input_shape: tuple(rows,cols),input_val_range: tuple(min,max),num_of_classes: int(num)},
#                       loss:{meta: {file_id: str(id), loss_type: str(type)}},
#                       optimizer:{meta: {file_id: str(id), optimizer_type: str(type)}}
#                       },
#               dataloader:{
#                       meta:{file_id: str(id), data_loader_type: str(type)}
#                       },
#               requirements.txt: {meta:{file_id: str(id)
#                 }
#               },
#               authentication:{
#                               bucket_name: str(name), access_key_id: str(id),
#                               secret_access_key: str(key),region: str(region)
#                       }
#             }
#
#           Example:
#                 {
#                   "ML_model": {
#                     "meta": {
#                       "file_id": "ML_file",
#                       "model_type": "sklearn",
#                       "ML_type": "classification",
#                       "algorithm": "randomForest"
#                     },
#                     "input": {
#                       "input_shape": "(100,20)",
#                       "input_val_range": "(10,20)"
#                       "num_of_classes": None
#                     },
#                     "loss": {
#                       "meta": {
#                         "file_id": "None",
#                         "loss_type": "None"
#                       }
#                     },
#                     "optimizer": {
#                       "meta": {
#                         "file_id": "None",
#                         "optimizer_type": "None"
#                       }
#                     }
#                   },
#                   "dataloader": {
#                     "meta": {
#                       "file_id": "Dataloader_file",
#                       "data_loader_type": "pandad.DataFrame"
#                     }
#                   },
#                   "requirements.txt": {
#                     "meta": {
#                       "file_id": "requirements.txt.txt"
#                     }
#
#                   },
#                   "authentication": {
#                   "bucket_name": "MyBucket",
#                   "access_key_id": "2k4k2jU992",
#                   "secret_access_key": "882LPh87",
#                   "region": "us.central"
#                   }
#                 }
#
#
#
#         Methods
#         ----------
#         All the methods are getter, setters and one method of that open connection.
#         Getters dwonloading files from S3 bucket, while setters uploading files to S3.
#
#
#
#         """
#         if isinstance(meta_data, str):
#             meta_data = json.loads(meta_data)
#         if isinstance(meta_data, dict):
#             self.__bucket = meta_data['authentication']['bucket_name']
#             self.__access_key_id = meta_data['authentication']['access_key_id']
#             self.__secret_access_key = meta_data['authentication']['secret_access_key']
#             self.__region = meta_data['authentication']['region']
#             self.__ML_model_file_id = meta_data['ML_model']['meta']['file_id']
#             self.__loss_function_file_id = meta_data['ML_model']['loss']['meta']['file_id']
#             self.__optimizer_file_id = meta_data['ML_model']['optimizer']['meta']['file_id']
#             self.__dataloader_file_id = meta_data['dataloader']['meta']['file_id']
#             self.__requirements_file_id = meta_data['requirements.txt']['meta']['file_id']
#         else:
#             raise TypeError('meta data need to be type dict or str')
#
#     def __connect(self):
#         """
#         Opens a connection with aws
#         """
#         session = boto3.Session(
#             aws_access_key_id=self.__access_key_id,
#             aws_secret_access_key=self.__secret_access_key,
#             region_name=self.__region,
#         )
#         return session
#
#     # Getters starts here
#     # All getters methods are downloading files from S3 bucket.
#
#     def get_model(self):
#         try:
#             session = self.__connect()
#             s3 = session.resource('s3')
#             model = pickle.loads(
#                 s3.Bucket(self.__bucket).Object(f"{self.__ML_model_file_id}.pickle").get()['Body'].read())
#             return model
#         except Exception as err:
#             logging.error(f"Downloading ML model from S3 failed!\nError:\n{err}")
#
#     def get_loss(self):
#         try:
#             if self.__loss_function_file_id:
#                 session = self.__connect()
#                 s3 = session.resource('s3')
#                 loss = pickle.loads(
#                     s3.Bucket(self.__bucket).Object(f"{self.__loss_function_file_id}.pickle").get()['Body'].read())
#                 return loss
#             logging.info('Loss function not provided')
#             return
#         except Exception as err:
#             logging.error(f"Downloading loss function from S3 failed!\nError:\n{err}")
#
#     def get_optimizer(self):
#         try:
#             if self.__optimizer_file_id:
#                 session = self.__connect()
#                 s3 = session.resource('s3')
#                 optimizer = pickle.loads(
#                     s3.Bucket(self.__bucket).Object(f"{self.__optimizer_file_id}.pickle").get()['Body'].read())
#                 return optimizer
#             logging.info('optimizer not provided')
#             return
#         except Exception as err:
#             logging.error(f"Downloading optimizer from S3 failed!\nError:\n{err}")
#
#     def get_dataloader(self):
#         try:
#             session = self.__connect()
#             s3 = session.resource('s3')
#             dataloader = pickle.loads(
#                 s3.Bucket(self.__bucket).Object(f"{self.__dataloader_file_id}.pickle").get()['Body'].read())
#             return dataloader
#         except Exception as err:
#             logging.error(f"Downloading dataloader from S3 failed!\nError:\n{err}")
#
#     def get_requirements(self):
#         try:
#             conn = self.__connect()
#             s3 = conn.resource('s3')
#             file_name = "requirements.txt.txt"
#             s3.Bucket(self.__bucket).download_file(self.__requirements_file_id,
#                                                    self.__requirements_file_id)
#             with open('requirements.txt.txt', 'r') as f:
#                 return f.read()
#         except Exception as err:
#             logging.error(f"Downloading requirements.txt file from S3 failed!\nError:\n{err}")
#
#     # Setters starts here
#     def upload(self, obj, obj_type):
#         hashmap = {"ML model": self.__ML_model_file_id,
#                    "loss": self.__loss_function_file_id,
#                    "optimizer": self.__optimizer_file_id,
#                    "dataloader": self.__dataloader_file_id,
#                    "requirements.txt": self.__requirements_file_id}
#         try:
#             logging.info(f'Uploading {obj_type} to S3 starts...')
#             logging.info('Connecting to S3...')
#             session = self.__connect()
#             s3 = session.resource('s3')
#             logging.info(f'Dumping {obj_type} to pickle file...')
#             binary_file = pickle.dumps(obj)
#             logging.info('Putting the model in S3....')
#             s3.Bucket(self.__bucket).put_object(Key=f'{hashmap[obj_type]}.pickle', Body=binary_file)
#             logging.info('Upload was successful!')
#         except Exception as err:
#             logging.error(f"Uploading {obj_type} to S3 failed!\nError:\n{err}")
#
#     def upload_requirements(self):
#         try:
#             logging.info('Uploading rquierments to S3 starts...')
#             logging.info('Connecting to S3...')
#             conn = self.__connect()
#             s3 = conn.resource('s3')
#             s3.Bucket(self.__bucket).upload_file(self.__requirements_file_id,
#                                                  self.__requirements_file_id)
#             logging.info('Upload was successful!')
#         except Exception as err:
#             logging.error(f"Uploading rquierments file to S3 failed!\nError:\n{err}")
#

class Bucket_loader:
    def __init__(self, meta_data,model=None, dataloader=None, loss=None, optimizer=None):
        """
        The bucket loader is an object that connect with aws and preform downloads
        and uploads from S3 bucket.

        Parameters
        ----------
                {valid: boolean,
                 ML_model:{
                      meta:{model_type: str(type), ML_type: str(type),algorithm: str(algorithm)},
                       input:{input_shape: tuple(rows,cols),input_val_range: tuple(min,max),num_of_classes: int(num)},
                      loss:{meta: {loss_type: boolean}},
                      optimizer:{meta: {optimizer_type: boolean}}
                      },
              dataloader:{
                      meta:{data_loader_type: str(type)}
                      }
                }

          Example:
                {valid: True,
                  "ML_model": {
                    "meta": {
                      "model_type": pytorch,
                      "ML_type": "classification",
                      "algorithm": None
                    },
                    "input": {
                      "input_shape": False,
                      "input_val_range": False,
                      "num_of_classes": True
                    },
                    "loss": {
                      "meta": {
                        "loss_type": True
                      }
                    },
                    "optimizer": {
                      "meta": {
                        "optimizer_type": False
                      }
                    }
                  },
                  "dataloader": {
                    "meta": {
                      "data_loader_type": "pandad.DataFrame"
                    }
                  }
                }

        meta_data : str\dict
            The json from user containig all meta data and structure as above.

        """
        self.model = model
        self.dataloader = dataloader
        self.loss = loss
        self.optimizer = optimizer

        self.__requirements_file_id = 'requirements.txt'

    # Getters starts here
    # All getters methods are downloading files from S3 bucket.

    def get_model(self):
        return self.model

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_dataloader(self):
        return self.dataloader

    def get_requirements(self):
        with open(self.__requirements_file_id, 'r') as f:
            return f.read()

    def upload(self, obj, obj_type):
        print(f"Uploading obj: {obj} with type: {obj_type}")
