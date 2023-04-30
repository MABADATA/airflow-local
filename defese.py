import random
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
from airflow.operators.python_operator import BranchPythonOperator
from utils.helpers import load_from_bucket, get_data
default_args = {
    'start_date': datetime(2021, 1, 1)
}
# ACOUNT_SERVICE_KEY=r"C:\Users\Administrator\Desktop\mabadata-733abc189d01.json"
# BUCKET_NAME='mabdata207125196'
json_file = {'Defence1': True, "Defence2": False, "Defence3": True}

def defence(ti):
    data = get_data()
    data_obj_type = str(type(data))
    ti.xcom_push(key='data_type:', value=data_obj_type)
    print('defensing')

def to_attack(ti):
    num_cycles = random.randint(1, 10)
    cycles = load_from_bucket('counter.pickle')
    if isinstance(cycles,str):
        ti.xcom_push(key='cycles', value=cycles)

    ti.xcom_push(key='cycles', value=str(cycles))
    cycles = num_cycles
    # ti.xcom_push(key='num_cycles', value=num_cycles)

    if isinstance(cycles, int) and cycles < 5:
        return 'trigger_offence'
    elif num_cycles > 5:
        return 'trigger_offence'
    else:
        return 'stop'

def stop():
    return 'Cycle has ended'
with DAG('defence_dag2',
    schedule_interval='@daily',
    default_args=default_args,
    catchup=False) as dag:

    storing = BashOperator(
        task_id='storing',
        bash_command='sleep 1'
    )

    defending = PythonOperator(
        task_id='defence1',
        python_callable=defence
    )

    branch = BranchPythonOperator(
        task_id='to_attack',
        python_callable=to_attack
    )

    stop_cycle = PythonOperator(
        task_id='stop',
        python_callable=stop
    )
    trigger_offence = TriggerDagRunOperator(
        task_id='trigger_offence',
        trigger_dag_id='attack_dag2'
    )


    storing >> defending >> branch >> [trigger_offence, stop_cycle]