# import random
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.operators.python_operator import BranchPythonOperator
# from datetime import datetime
# from airflow.operators.trigger_dagrun import TriggerDagRunOperator
# from utils.helpers import load_from_bucket, upload_to_bucket
# from sklearn.metrics import accuracy_score
# import numpy as np
# from utils.defence_helpers import pars_defence_json
# from utils.defence_check import *
# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'start_date': datetime(2023, 3, 16),
#     'retries': 1
# }
#
#
# def run_function(func_name, run_func):
#     if run_func:
#         # Run the function
#         return f"Running function: {func_name}"
#     else:
#         return f"Not running function: {func_name}"
#
# def defender_turn(defense, classifier, adv_examples, y_test):
#     classifier.postprocessing_defences.append(defense)  # each time we will add the next defense on top of the last one.
#     prediction_softmax_results_def = classifier.predict(adv_examples)
#     prediction_results_def = np.argmax(prediction_softmax_results_def, axis=1)
#     # get the model's accuracy after the denfese applied and return it.
#     return accuracy_score(y_test, prediction_results_def)
#
# def defence_HighConfidence(ti):
#     random_score = random.randint(1,10)
#     ti.xcom_push(key='HighConfidence_score', value=random_score)
#
# def defence_ReverseSigmoid(ti):
#     random_score = random.randint(1, 10)
#     ti.xcom_push(key='ReverseSigmoid_score', value=random_score)
#
# def defence_Rounded(ti):
#     random_score = random.randint(1, 10)
#     ti.xcom_push(key='Rounded_score', value=random_score)
#
# def defence_GaussianNoise(ti):
#     random_score = random.randint(1, 10)
#     ti.xcom_push(key='GaussianNoise_score', value=random_score)
#
# def default(ti):
#     ti.xcom_push(key='Default', value=0)
#
# def choose_best_defence(ti):
#     defences_scores = {'Defence1_score':0,'Defence2_score':0,'Defence3_score':0}
#     for index,attacks_score in enumerate(defences_scores):
#         score = ti.xcom_pull(key=attacks_score,
#                              task_ids=f'Defence{index + 1}')
#         if score:
#             defences_scores[attacks_score] = ti.xcom_pull(key=attacks_score,
#                                  task_ids=f'Defence{index + 1}')
#
#     best_score = max(defences_scores.values())
#     metadata = load_from_bucket(file_name='attack_defence_metadata.json', as_json=True)
#     for key, val in defences_scores.items():
#         if val == best_score:
#             metadata['cycles'] += 1
#             metadata['defence_best_scores'].append((key,val))
#             ti.xcom_push(key=f'best defence {key}', value=val)
#             upload_to_bucket(obj=metadata, file_name='attack_defence_metadata.json', as_json=True)
#             return
#
# def to_defend(ti):
#     cycles = load_from_bucket('attack_defence_metadata.json',as_json=True)['cycles']
#     # num_of_cycles_user_request = load_from_bucket('user_metadata.json',as_json=True)['cycles']
#     num_of_cycles_user_request = 5
#     if isinstance(cycles, int) and cycles < num_of_cycles_user_request:
#         return 'trigger_offence'
#     else:
#         return 'stop'
#
# def stop():
#     return 'Cycle has ended'
#
#
# with DAG('multi_defence_dag', schedule_interval='@daily', default_args=default_args, catchup=False) as dag:
#
#     choose_defence = PythonOperator(
#             task_id=f'choose_defence',
#             python_callable=pars_defence_json
#         )
#
#     branch_HighConfidence = BranchPythonOperator(
#         task_id='to_defence_HighConfidence',
#         python_callable=to_defence_HighConfidence
#     )
#     branch_ReverseSigmoid = BranchPythonOperator(
#         task_id='to_defence_ReverseSigmoid',
#         python_callable=to_defence_ReverseSigmoid
#     )
#     branch_Rounded = BranchPythonOperator(
#         task_id='to_defence_Rounded',
#         python_callable=to_defence_Rounded
#     )
#
#     branch_GaussianNoise = BranchPythonOperator(
#         task_id='to_defence_GaussianNoise',
#         python_callable=to_defence_GaussianNoise
#     )
#
#     run_defence_ReverseSigmoid = PythonOperator(
#             task_id=f"defence_ReverseSigmoid",
#             python_callable=defence_ReverseSigmoid
#         )
#     run_defence_HighConfidence = PythonOperator(
#             task_id=f"defence_HighConfidence",
#             python_callable=defence_HighConfidence
#         )
#     run_defence_Rounded = PythonOperator(
#             task_id=f"defence_Rounded",
#             python_callable=defence_Rounded
#         )
#     run_defence_GaussianNoise = PythonOperator(
#             task_id=f"defence_GaussianNoise",
#             python_callable=defence_GaussianNoise
#         )
#     run_default = PythonOperator(
#             task_id=f"default",
#             python_callable=default
#         )
#     choose_best = PythonOperator(
#             task_id=f"choose_best",
#             python_callable=choose_best_defence,
#             trigger_rule='none_failed'
#         )
#     trigger_branch = BranchPythonOperator(
#         task_id='to_defend',
#         python_callable=to_defend
#     )
#
#     stop_cycle = PythonOperator(
#         task_id='stop',
#         python_callable=stop
#     )
#     trigger_offence = TriggerDagRunOperator(
#         task_id='trigger_offence',
#         trigger_dag_id='multi_attack_dag'
#     )
#     #
#     choose_defence >> branch_Rounded >> [run_defence_Rounded, run_default] >> choose_best
#     choose_defence >> branch_GaussianNoise >> [run_defence_GaussianNoise, run_default] >> choose_best
#     choose_defence >> branch_HighConfidence >> [run_defence_HighConfidence, run_default] >> choose_best
#     choose_defence >> branch_ReverseSigmoid >> [run_defence_ReverseSigmoid, run_default] >> choose_best
#     choose_best >> trigger_branch >> [trigger_offence, stop_cycle]