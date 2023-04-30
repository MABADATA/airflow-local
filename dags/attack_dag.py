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
    # branch_SquareAttack = BranchPythonOperator(
    #     task_id='to_attack_SquareAttack',
    #     python_callable=to_attack_SquareAttack
    # )
    #
    # run_attack_SquareAttack = PythonOperator(
    #         task_id=f"attack_SquareAttack",
    #         python_callable=attack_SquareAttack)
    # branch_TargetedUniversalPerturbation = BranchPythonOperator(
    #     task_id='to_attack_TargetedUniversalPerturbation',
    #     python_callable=to_attack_TargetedUniversalPerturbation
    # )
    #
    # run_attack_TargetedUniversalPerturbation = PythonOperator(
    #         task_id=f"attack_TargetedUniversalPerturbation",
    #         python_callable=attack_TargetedUniversalPerturbation)
    branch_UniversalPerturbation = BranchPythonOperator(
        task_id='to_attack_UniversalPerturbation',
        python_callable=to_attack_UniversalPerturbation
    )

    run_attack_UniversalPerturbation = PythonOperator(
            task_id=f"attack_UniversalPerturbation",
            python_callable=attack_UniversalPerturbation)
    # branch_VirtualAdversarialMethod = BranchPythonOperator(
    #     task_id='to_attack_VirtualAdversarialMethod',
    #     python_callable=to_attack_VirtualAdversarialMethod
    # )
    #
    # run_attack_VirtualAdversarialMethod = PythonOperator(
    #         task_id=f"attack_VirtualAdversarialMethod",
    #         python_callable=attack_VirtualAdversarialMethod)
    # branch_Wasserstein = BranchPythonOperator(
    #     task_id='to_attack_Wasserstein',
    #     python_callable=to_attack_Wasserstein
    # )

    # run_attack_Wasserstein = PythonOperator(
    #         task_id=f"attack_Wasserstein",
    #         python_callable=attack_Wasserstein)
    # branch_ZooAttack = BranchPythonOperator(
    #     task_id='to_attack_ZooAttack',
    #     python_callable=to_attack_ZooAttack
    # )

    # run_attack_ZooAttack = PythonOperator(
    #         task_id=f"attack_ZooAttack",
    #         python_callable=attack_ZooAttack)
    # branch_FrameSaliencyAttack = BranchPythonOperator(
    #     task_id='to_attack_FrameSaliencyAttack',
    #     python_callable=to_attack_FrameSaliencyAttack
    # )

    # run_attack_FrameSaliencyAttack = PythonOperator(
    #         task_id=f"attack_FrameSaliencyAttack",
    #         python_callable=attack_FrameSaliencyAttack)
    # branch_GeoDA = BranchPythonOperator(
    #     task_id='to_attack_GeoDA',
    #     python_callable=to_attack_GeoDA
    # )

    # run_attack_GeoDA = PythonOperator(
    #         task_id=f"attack_GeoDA",
    #         python_callable=attack_GeoDA)
    # branch_ElasticNet = BranchPythonOperator(
    #     task_id='to_attack_ElasticNet',
    #     python_callable=to_attack_ElasticNet
    # )

    # run_attack_ElasticNet = PythonOperator(
    #         task_id=f"attack_ElasticNet",
    #         python_callable=attack_ElasticNet)
    branch_CarliniL2Method = BranchPythonOperator(
        task_id='to_attack_CarliniL2Method',
        python_callable=to_attack_CarliniL2Method
    )

    run_attack_CarliniL2Method = PythonOperator(
            task_id=f"attack_CarliniL2Method",
            python_callable=attack_CarliniL2Method)
    # branch_BoundaryAttack = BranchPythonOperator(
    #     task_id='to_attack_BoundaryAttack',
    #     python_callable=to_attack_BoundaryAttack
    # )

    # run_attack_BoundaryAttack = PythonOperator(
    #         task_id=f"attack_BoundaryAttack",
    #         python_callable=attack_BoundaryAttack)
    # branch_AutoProjectedGradientDescent = BranchPythonOperator(
    #     task_id='to_attack_AutoProjectedGradientDescent',
    #     python_callable=to_attack_AutoProjectedGradientDescent
    # )

    # run_attack_AutoProjectedGradientDescent = PythonOperator(
    #         task_id=f"attack_AutoProjectedGradientDescent",
    #         python_callable=attack_AutoProjectedGradientDescent)
    # branch_DeepFool = BranchPythonOperator(
    #     task_id='to_attack_DeepFool',
    #     python_callable=to_attack_DeepFool
    # )

    # run_attack_DeepFool = PythonOperator(
    #         task_id=f"attack_DeepFool",
    #         python_callable=attack_DeepFool)
    # branch_AutoAttack = BranchPythonOperator(
    #     task_id='to_attack_AutoAttack',
    #     python_callable=to_attack_AutoAttack
    # )

    # run_attack_AutoAttack = PythonOperator(
    #         task_id=f"attack_AutoAttack",
    #         python_callable=attack_AutoAttack)
    # branch_LowProFool = BranchPythonOperator(
    #     task_id='to_attack_LowProFool',
    #     python_callable=to_attack_LowProFool
    # )

    # run_attack_LowProFool = PythonOperator(
    #         task_id=f"attack_LowProFool",
    #         python_callable=attack_LowProFool)
    branch_NewtonFool = BranchPythonOperator(
        task_id='to_attack_NewtonFool',
        python_callable=to_attack_NewtonFool
    )

    run_attack_NewtonFool = PythonOperator(
            task_id=f"attack_NewtonFool",
            python_callable=attack_NewtonFool)
    # branch_MalwareGDTensorFlow = BranchPythonOperator(
    #     task_id='to_attack_MalwareGDTensorFlow',
    #     python_callable=to_attack_MalwareGDTensorFlow
    # )

    # run_attack_MalwareGDTensorFlow = PythonOperator(
    #         task_id=f"attack_MalwareGDTensorFlow",
    #         python_callable=attack_MalwareGDTensorFlow)
    # branch_PixelAttack = BranchPythonOperator(
    #     task_id='to_attack_PixelAttack',
    #     python_callable=to_attack_PixelAttack
    # )

    # run_attack_PixelAttack = PythonOperator(
    #         task_id=f"attack_PixelAttack",
    #         python_callable=attack_PixelAttack)
    # branch_SaliencyMapMethod = BranchPythonOperator(
    #     task_id='to_attack_SaliencyMapMethod',
    #     python_callable=to_attack_SaliencyMapMethod
    # )

    # run_attack_SaliencyMapMethod = PythonOperator(
    #         task_id=f"attack_SaliencyMapMethod",
    #         python_callable=attack_SaliencyMapMethod)
    # branch_ShadowAttack = BranchPythonOperator(
    #     task_id='to_attack_ShadowAttack',
    #     python_callable=to_attack_ShadowAttack
    # )
    #
    # run_attack_ShadowAttack = PythonOperator(
    #         task_id=f"attack_ShadowAttack",
    #         python_callable=attack_ShadowAttack)
    # branch_SpatialTransformation = BranchPythonOperator(
    #     task_id='to_attack_SpatialTransformation',
    #     python_callable=to_attack_SpatialTransformation
    # )
    #
    # run_attack_SpatialTransformation = PythonOperator(
    #         task_id=f"attack_SpatialTransformation",
    #         python_callable=attack_SpatialTransformation)
    # branch_ShapeShifter = BranchPythonOperator(
    #     task_id='to_attack_ShapeShifter',
    #     python_callable=to_attack_ShapeShifter
    # )
    #
    # run_attack_ShapeShifter = PythonOperator(
    #         task_id=f"attack_ShapeShifter",
    #         python_callable=attack_ShapeShifter)
    # branch_SignOPTAttack = BranchPythonOperator(
    #     task_id='to_attack_SignOPTAttack',
    #     python_callable=to_attack_SignOPTAttack
    # )
    #
    # run_attack_SignOPTAttack = PythonOperator(
    #         task_id=f"attack_SignOPTAttack",
    #         python_callable=attack_SignOPTAttack)
    # branch_AdversarialPatch = BranchPythonOperator(
    #     task_id='to_attack_AdversarialPatch',
    #     python_callable=to_attack_AdversarialPatch
    # )
    #
    # run_attack_AdversarialPatch = PythonOperator(
    #         task_id=f"attack_AdversarialPatch",
    #         python_callable=attack_AdversarialPatch)
    # branch_AdversarialPatchPyTorch = BranchPythonOperator(
    #     task_id='to_attack_AdversarialPatchPyTorch',
    #     python_callable=to_attack_AdversarialPatchPyTorch
    # )
    #
    # run_attack_AdversarialPatchPyTorch = PythonOperator(
    #         task_id=f"attack_AdversarialPatchPyTorch",
    #         python_callable=attack_AdversarialPatchPyTorch)
    # branch_FeatureAdversariesPyTorch = BranchPythonOperator(
    #     task_id='to_attack_FeatureAdversariesPyTorch',
    #     python_callable=to_attack_FeatureAdversariesPyTorch
    # )
    #
    # run_attack_FeatureAdversariesPyTorch = PythonOperator(
    #         task_id=f"attack_FeatureAdversariesPyTorch",
    #         python_callable=attack_FeatureAdversariesPyTorch)
    # branch_GRAPHITEBlackbox = BranchPythonOperator(
    #     task_id='to_attack_GRAPHITEBlackbox',
    #     python_callable=to_attack_GRAPHITEBlackbox
    # )
    #
    # run_attack_GRAPHITEBlackbox = PythonOperator(
    #         task_id=f"attack_GRAPHITEBlackbox",
    #         python_callable=attack_GRAPHITEBlackbox)
    # branch_GRAPHITEWhiteboxPyTorch = BranchPythonOperator(
    #     task_id='to_attack_GRAPHITEWhiteboxPyTorch',
    #     python_callable=to_attack_GRAPHITEWhiteboxPyTorch
    # )
    #
    # run_attack_GRAPHITEWhiteboxPyTorch = PythonOperator(
    #         task_id=f"attack_GRAPHITEWhiteboxPyTorch",
    #         python_callable=attack_GRAPHITEWhiteboxPyTorch)
    # branch_LaserAttack = BranchPythonOperator(
    #     task_id='to_attack_LaserAttack',
    #     python_callable=to_attack_LaserAttack
    # )
    #
    # run_attack_LaserAttack = PythonOperator(
    #         task_id=f"attack_LaserAttack",
    #         python_callable=attack_LaserAttack)
    # branch_OverTheAirFlickeringPyTorch = BranchPythonOperator(
    #     task_id='to_attack_OverTheAirFlickeringPyTorch',
    #     python_callable=to_attack_OverTheAirFlickeringPyTorch
    # )
    #
    # run_attack_OverTheAirFlickeringPyTorch = PythonOperator(
    #         task_id=f"attack_OverTheAirFlickeringPyTorch",
    #         python_callable=attack_OverTheAirFlickeringPyTorch)
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
    #
    choose_attack >> metadata >> branch_BasicIterativeMethod >> [run_attack_BasicIterativeMethod, run_default] >> choose_best
    choose_attack >> metadata >> branch_FastGradientMethod >> [run_attack_FastGradientMethod, run_default] >> choose_best
    choose_attack >> metadata >> branch_ProjectedGradientDescent >> [run_attack_ProjectedGradientDescent, run_default] >> choose_best
    # choose_attack >> metadata >> branch_SquareAttack >> [run_attack_SquareAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_TargetedUniversalPerturbation >> [run_attack_TargetedUniversalPerturbation, run_default] >> choose_best
    choose_attack >> metadata >> branch_UniversalPerturbation >> [run_attack_UniversalPerturbation, run_default] >> choose_best
    # choose_attack >> metadata >> branch_VirtualAdversarialMethod >> [run_attack_VirtualAdversarialMethod, run_default] >> choose_best
    # choose_attack >> metadata >> branch_Wasserstein >> [run_attack_Wasserstein, run_default] >> choose_best
    # choose_attack >> metadata >> branch_ZooAttack >> [run_attack_ZooAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_FrameSaliencyAttack >> [run_attack_FrameSaliencyAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_GeoDA >> [run_attack_GeoDA, run_default] >> choose_best
    # choose_attack >> metadata >> branch_ElasticNet >> [run_attack_ElasticNet, run_default] >> choose_best
    choose_attack >> metadata >> branch_CarliniL2Method >> [run_attack_CarliniL2Method, run_default] >> choose_best
    # choose_attack >> metadata >> branch_BoundaryAttack >> [run_attack_BoundaryAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_AutoProjectedGradientDescent >> [run_attack_AutoProjectedGradientDescent, run_default] >> choose_best
    # choose_attack >> metadata >> branch_DeepFool >> [run_attack_DeepFool, run_default] >> choose_best
    # choose_attack >> metadata >> branch_AutoAttack >> [run_attack_AutoAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_LowProFool >> [run_attack_LowProFool, run_default] >> choose_best
    choose_attack >> metadata >> branch_NewtonFool >> [run_attack_NewtonFool, run_default] >> choose_best
    # choose_attack >> metadata >> branch_MalwareGDTensorFlow >> [run_attack_MalwareGDTensorFlow, run_default] >> choose_best
    # choose_attack >> metadata >> branch_PixelAttack >> [run_attack_PixelAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_SaliencyMapMethod >> [run_attack_SaliencyMapMethod, run_default] >> choose_best
    # choose_attack >> metadata >> branch_ShadowAttack >> [run_attack_ShadowAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_SpatialTransformation >> [run_attack_SpatialTransformation, run_default] >> choose_best
    # choose_attack >> metadata >> branch_ShapeShifter >> [run_attack_ShapeShifter, run_default] >> choose_best
    # choose_attack >> metadata >> branch_SignOPTAttack >> [run_attack_SignOPTAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_AdversarialPatch >> [run_attack_AdversarialPatch, run_default] >> choose_best
    # choose_attack >> metadata >> branch_AdversarialPatchPyTorch >> [run_attack_AdversarialPatchPyTorch, run_default] >> choose_best
    # choose_attack >> metadata >> branch_FeatureAdversariesPyTorch >> [run_attack_FeatureAdversariesPyTorch, run_default] >> choose_best
    # choose_attack >> metadata >> branch_GRAPHITEBlackbox >> [run_attack_GRAPHITEBlackbox, run_default] >> choose_best
    # choose_attack >> metadata >> branch_GRAPHITEWhiteboxPyTorch >> [run_attack_GRAPHITEWhiteboxPyTorch, run_default] >> choose_best
    # choose_attack >> metadata >> branch_LaserAttack >> [run_attack_LaserAttack, run_default] >> choose_best
    # choose_attack >> metadata >> branch_OverTheAirFlickeringPyTorch >> [run_attack_OverTheAirFlickeringPyTorch, run_default] >> choose_best
    choose_best >> trigger_defence
#

