from datetime import datetime
import json
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from utils.helpers import *
from art.attacks.evasion import *
from art.attacks.poisoning import *
from art.attacks.inference import *
from art.attacks.extraction import *
import logging
def generate_dynamic_dag(ti):

    # json_data = load_from_bucket("attack_defence_metadata.json",as_json=True)
    json_data = {"attacks": [ZooAttack.__name__,ProjectedGradientDescent.__name__]}
    attacks = json_data['attacks']
    for attack in attacks:
        add_attack(attack_name=attack,attack_dag_file_name="dags/attack_dag.py",attack_file_name='dags/utils/attack.py')
    #
    # defences = json_data['defences']
    # for defence in defences:
    #     add_attack(attack_name=defence)
#
dag = DAG(
    'attack_defence_dags_gen',
    description='Dynamically generate DAGs',
    schedule_interval=None,
    start_date=datetime(2023, 5, 23),
    catchup=False
)

with dag:
    generate_dag_task = PythonOperator(
        task_id='generate_dynamic_dag',
        python_callable=generate_dynamic_dag,
        op_kwargs={'json_data': '{"param": "value"}'}
    )

    trigger_dag_task = TriggerDagRunOperator(
        task_id='trigger_attack_dag',
        trigger_dag_id='multi_attack_dag'
    )

    generate_dag_task >> trigger_dag_task
