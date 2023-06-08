import os
import logging
import json
from file_loader.file_handler import FileLoader
from user_files.helpers import get_files_package_root
class Envsetter:
  """
  This class purpose is to set the environment to be same to user's environment
  """
  def __init__(self, metadata) -> None:
    """
     Parameters
        ----------
        requirements_file_path : str
            The path of the requirements.txt.txt file
        """
    self.__file_loader = FileLoader(metadata)
    self.__requirements_file_id = metadata['requirements.txt']['meta']['file_id']
    self.__account_service_key = metadata['authentication']['account_service_key']
    self.__ML_model_file_id = metadata['ML_model']['meta']['file_id']
    self.__ML_model_script_file_id = metadata['ML_model']['meta']['script_file_id']
    self.__ML_model_class_name = metadata['ML_model']['meta']['class_name']
    os.environ["BUCKET_NAME"] = metadata['authentication']['bucket_name']
    os.environ["ACOUNT_SERVICE_KEY"] = "account_service_key.json"
    with open("account_service_key.json", 'w') as f:
      json.dump(self.__account_service_key, f)

  def install_requirements(self):
    # Use the 'pip' command to install the packages listed in the requirements.txt file
    try:
      self.__file_loader.get_file(self.__requirements_file_id)
      requirements_loc =  get_files_package_root() + "/" + self.__requirements_file_id
      os.system(f"pip install -r {requirements_loc}")
      logging.info("Environment is set! requirements.txt installed successfully!")
    except Exception as err:
      logging.error(f"Exception raised during requirements.txt installation!\nException:\n{err}")
