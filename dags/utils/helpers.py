import os
import cloudpickle
import pickle
import numpy as np
import sklearn.model_selection
from gcloud import storage
import json
import re
import pandas as pd
import json
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

CLASSIFIER_DICT = {
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
                    "ScikitlearnDecisionTreeRegressor":  ScikitlearnDecisionTreeRegressor
}




def get_client():
    ACOUNT_SERVICE_KEY = os.environ.get('ACCOUNT_SERVICE_KEY')
    BUCKET_NAME = os.environ.get('BUCKET_NAME')
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACOUNT_SERVICE_KEY
    os.environ["DONT_PICKLE"] = 'False'
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket

# def get_data():
#
#     features_file_name = 'X.csv'
#     label_file_name = 'y.csv'
#     data_size = 1000
#     bucket = get_client()
#     try:
#         #get X - features set
#         blob = bucket.blob(features_file_name)
#         blob.download_to_filename(features_file_name)
#         X = pd.read_csv(features_file_name)[:data_size]
#         # get y - label set
#         blob = bucket.blob(label_file_name)
#         blob.download_to_filename(label_file_name)
#         y = pd.read_csv(label_file_name)[:data_size]
#         # only for this data set
#         x_test = X.label
#         x_train = X.drop('label', axis=1)
#         y_test = y.label
#         y_train = y.drop('label', axis=1)
#         # x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
#         data = [(x_train,x_test),(y_train,y_test)]
#         for i in range(len(data)):
#             yield data[i]
#     except Exception as err:
#         return
def get_data():
    #need to get x_train,y_train for estimator fit and x_test, y_test for acc
    train_features_file_name = 'X_train.csv'
    test_features_file_name = 'X_test.csv'
    train_label_file_name = 'y_train.csv'
    test_label_file_name = 'y_test.csv'
    data_size = -1
    bucket = get_client()
    try:
        #get X_train - train features set
        blob = bucket.blob(train_features_file_name)
        blob.download_to_filename(train_features_file_name)
        X_train = pd.read_csv(train_features_file_name)[:data_size].to_numpy()
        #get X_test - test features set
        blob = bucket.blob(test_features_file_name)
        blob.download_to_filename(test_features_file_name)
        X_test = pd.read_csv(test_features_file_name)[:data_size].to_numpy()
        # get y_train - train label set
        blob = bucket.blob(train_label_file_name)
        blob.download_to_filename(train_label_file_name)
        y_train = pd.read_csv(train_label_file_name)[:data_size].to_numpy()
        # get y_test - train label set
        blob = bucket.blob(test_label_file_name)
        blob.download_to_filename(test_label_file_name)
        y_test = pd.read_csv(test_label_file_name)[:data_size].to_numpy()
        data = [(X_train,y_train),(X_test,y_test)]
        return data
    except Exception as err:
        return
def load_from_bucket(file_name, as_json=False,as_csv=False):

    bucket = get_client()
    try:
        blob = bucket.blob(file_name)
        if as_json:
            json_string = blob.download_as_string()
            data = json.loads(json_string)
            return data
        elif as_csv:
            blob.download_to_filename(file_name)
            df = pd.read_csv(file_name, dtype=float)
            np_array = np.array(df)
            return np_array
        else:
            blob.download_to_filename(file_name)
            with open(file_name, 'rb') as f:
                loaded_obj = cloudpickle.load(f)
                return loaded_obj
    except Exception as err:
        raise err


def upload_to_bucket(obj,file_name,as_json=False,as_csv=False):
    bucket = get_client()
    blob = bucket.blob(file_name)


    if as_json:
        json_file = json.dumps(obj)
        blob.upload_from_string(json_file)
    elif as_csv:
        # obj_as_nparray = np.array(obj)
        pd.DataFrame(obj).to_csv(file_name, index=False)
        blob.upload_from_filename(file_name)
    else:
        with open(file_name, 'wb') as f:
            cloudpickle.dump(obj, f, protocol=4)
            blob.upload_from_filename(file_name)

def get_estimator_object(classifier_name):
    return CLASSIFIER_DICT[classifier_name]

def add_attack(attack_name,attack_file_name="dags/attack.py",
               attack_check_file_name="dags/attack_check.py",attack_dag_file_name="dags/attack_dag.py"):

    def add_attack_inner(script, indent_pos, re_express,
               file_name):
        with open(f'{file_name}', 'r') as f:
            # Read the contents of the file into a list of lines
            lines = f.readlines()

        attack_pattern = re.compile(rf"{re_express}")
        my_line = None
        for i, line in enumerate(lines):
            if attack_pattern.search(line):
                my_line = i
                break

        indent = '\t' * indent_pos
        lines.insert(my_line, f"{indent}{script}\n")
        with open(f'{file_name}', 'w') as f:
            f.writelines(lines)

    # scripts to add to the file
    script_to_attack = f"""def to_attack_{attack_name}(ti):
    to_attack = ti.xcom_pull(key='attack_{attack_name}',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_{attack_name}'
    else:
        return 'default'"""

    attack_script = f"""def attack_{attack_name}(ti):
    model_acc, adversarial_examples  = attack({attack_name})
    ti.xcom_push(key='attack_{attack_name}_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_{attack_name}_score', value=model_acc)
"""
    operator_script = f"""    branch_{attack_name} = BranchPythonOperator(
        task_id='to_attack_{attack_name}',
        python_callable=to_attack_{attack_name}
    )

    run_attack_{attack_name} = PythonOperator(
            task_id=f"attack_{attack_name}",
            python_callable=attack_{attack_name})"""

    dag_script = f"    choose_attack >> metadata >> branch_{attack_name} >> [run_attack_{attack_name}, run_default] >> choose_best"
    #
    insert_param = {"to_attack":{"script":script_to_attack ,"indent": 0, "re": "^\s*def\s+to_attack_BasicIterativeMethod\s*\(","file_name":f"{attack_check_file_name}"},
                    "attack":{"script":attack_script,"indent": 0, "re": "^\s*def\s+attack_BasicIterativeMethod\s*\(","file_name":f"{attack_file_name}"},
                    "operator":{"script":operator_script,"indent": 0, "re": "^\s+branch_BasicIterativeMethod\s*","file_name":f"{attack_dag_file_name}" },
                    "DAG_line":{"script":dag_script,"indent": 0, "re": "^\s+choose_best\s*>>\s*","file_name":f"{attack_dag_file_name}"} }

    for key, params in insert_param.items():
        add_attack_inner(script=params['script'],indent_pos=params['indent'],re_express=params['re'],file_name=params['file_name'])
