from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from datetime import timedelta
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
from utils.attack import *
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 16),
    'retries': 1
}

def default_attack(ti):
    ti.xcom_push(key='Default', value=0)
    with open('sample_file.json', 'r') as f:
        file = json.load(f)
        ti.xcom_push(key='json', value=file)



with DAG('multi_attack_dag', schedule_interval='@daily', default_args=default_args, catchup=False,dagrun_timeout=timedelta(minutes=240)) as dag:


    metadata = PythonOperator(
        task_id="metadata_prep",
        python_callable=set_or_create
    )
    run_default = PythonOperator(
            task_id=f"default",
            python_callable=default_attack
        )


    run_attack_ZooAttack = PythonOperator(
            task_id=f"attack_ZooAttack",
            python_callable=attack_ZooAttack)
    run_attack_ProjectedGradientDescent = PythonOperator(
            task_id=f"attack_ProjectedGradientDescent",
            python_callable=attack_ProjectedGradientDescent)
    #Python operator place

    trigger_defence = TriggerDagRunOperator(
        task_id='trigger_offence',
        trigger_dag_id='multi_defence_dag'
    )
    choose_best = PythonOperator(
            task_id=f"choose_best",
            python_callable=choose_best_attack,
            trigger_rule='none_failed'
        )

    metadata >> [run_attack_ZooAttack, run_default] >> choose_best
    metadata >> [run_attack_ProjectedGradientDescent, run_default] >> choose_best
    choose_best >> trigger_defence


