from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from datetime import timedelta
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime
from utils.attack import *
from utils.attack_check import *
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 16),
    'retries': 1
}

def default_attack(ti):
    ti.xcom_push(key='Default', value=0)

with DAG('multi_attack_dag', schedule_interval='@daily', default_args=default_args, catchup=False,dagrun_timeout=timedelta(minutes=240)) as dag:

    choose_attack = PythonOperator(
            task_id=f'choose_attack',
            python_callable=pars_attack_json
        )

    metadata = PythonOperator(
        task_id="metadata",
        python_callable=set_or_create
    )
    branch_UniversalPerturbation = BranchPythonOperator(
        task_id='to_attack_UniversalPerturbation',
        python_callable=to_attack_UniversalPerturbation
    )

    run_attack_UniversalPerturbation = PythonOperator(
            task_id=f"attack_UniversalPerturbation",
            python_callable=attack_UniversalPerturbation)

    branch_CarliniL2Method = BranchPythonOperator(
        task_id='to_attack_CarliniL2Method',
        python_callable=to_attack_CarliniL2Method
    )

    run_attack_CarliniL2Method = PythonOperator(
            task_id=f"attack_CarliniL2Method",
            python_callable=attack_CarliniL2Method)

    branch_NewtonFool = BranchPythonOperator(
        task_id='to_attack_NewtonFool',
        python_callable=to_attack_NewtonFool
    )

    run_attack_NewtonFool = PythonOperator(
            task_id=f"attack_NewtonFool",
            python_callable=attack_NewtonFool)

    branch_BasicIterativeMethod = BranchPythonOperator(
        task_id='to_attack_BasicIterativeMethod',
        python_callable=to_attack_BasicIterativeMethod
    )
    branch_FastGradientMethod = BranchPythonOperator(
        task_id='to_attack_FastGradientMethod',
        python_callable=to_attack_FastGradientMethod
    )
    branch_ProjectedGradientDescent = BranchPythonOperator(
        task_id='to_attack_ProjectedGradientDescent',
        python_callable=to_attack_ProjectedGradientDescent
    )
    #
    run_attack_BasicIterativeMethod = PythonOperator(
            task_id=f"attack_BasicIterativeMethod",
            python_callable=attack_BasicIterativeMethod
        )
    run_attack_FastGradientMethod = PythonOperator(
            task_id=f"attack_FastGradientMethod",
            python_callable=attack_FastGradientMethod
        )
    run_attack_ProjectedGradientDescent = PythonOperator(
            task_id=f"attack_ProjectedGradientDescent",
            python_callable=attack_ProjectedGradientDescent
        )
    run_default = PythonOperator(
            task_id=f"default_attack",
            python_callable=default_attack
        )
    choose_best = PythonOperator(
            task_id=f"choose_best",
            python_callable=choose_best_attack,
            trigger_rule='all_done'
        )
    trigger_defence = TriggerDagRunOperator(
        task_id='trigger_defence',
        trigger_dag_id='multi_defence_dag'
    )
    choose_attack >> metadata >> branch_BasicIterativeMethod >> [run_attack_BasicIterativeMethod, run_default] >> choose_best
    choose_attack >> metadata >> branch_FastGradientMethod >> [run_attack_FastGradientMethod, run_default] >> choose_best
    choose_attack >> metadata >> branch_ProjectedGradientDescent >> [run_attack_ProjectedGradientDescent, run_default] >> choose_best
    choose_attack >> metadata >> branch_UniversalPerturbation >> [run_attack_UniversalPerturbation, run_default] >> choose_best
    choose_attack >> metadata >> branch_CarliniL2Method >> [run_attack_CarliniL2Method, run_default] >> choose_best
    choose_attack >> metadata >> branch_NewtonFool >> [run_attack_NewtonFool, run_default] >> choose_best
    choose_best >> trigger_defence


