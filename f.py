import random
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from datetime import datetime,timedelta
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from helpers import load_from_bucket, upload_to_bucket, get_data
import numpy as np
from sklearn.metrics import accuracy_score
from art.attacks.evasion import BasicIterativeMethod, FastGradientMethod,ProjectedGradientDescent,auto_projected_gradient_descent
import json
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression.pytorch import PyTorchRegressor
from art.estimators.classification import TensorFlowClassifier, TensorFlowV2Classifier
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
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 16),
    'retries': 1
}
def assign_vars(cls, args_dict,ML_model):
    if args_dict.get("optimizer"):
        optimizer = SGD(ML_model.parameters(), lr=0.01)
        args_dict["optimizer"] = optimizer
    if args_dict.get("loss"):
        loss = load_from_bucket('loss.pickle')
        # only from test
        loss = CrossEntropyLoss()
        args_dict["loss"] = loss
    obj = cls(**args_dict,model=ML_model)
    return obj
def get_obj_from_str(obj_as_str):
    estimator_map = {
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
        "ScikitlearnDecisionTreeRegressor": ScikitlearnDecisionTreeRegressor
    }
    return estimator_map[obj_as_str]

def get_estimator():
    args = load_from_bucket("Estimator_params.json",as_json=True)
    model = load_from_bucket("ML_model.pickle",as_json=False)
    estimator_str = args['object']
    estimator_obj = get_obj_from_str(estimator_str)
    params = args['params']
    estimator = assign_vars(cls=estimator_obj,args_dict=params,ML_model=model)
    return estimator

def sent_model_after_attack(estimator):
    upload_to_bucket(estimator.model, 'ML_model.pickle')

def run_function(func_name, run_func):
    if run_func:
        # Run the function
        return f"Running function: {func_name}"
    else:
        return f"Not running function: {func_name}"


def pars_json(ti):
    # json_data = get_json_from_bucket()
    # attack_json = json.loads(load_from_bucket('attack.json'))
    attack_json = {'attack_BasicIterativeMethod': True,
                   'attack_FastGradientMethod': False, 'attack_ProjectedGradientDescent': True}
    for attack, bool_val in attack_json.items():
        ti.xcom_push(key=attack, value=bool_val)





def to_attack_auto_projected_gradient_descent(ti):
    to_attack = ti.xcom_pull(key='attack_auto_projected_gradient_descent',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_auto_projected_gradient_descent'
    else:
        return 'default'


def to_attack_SquareAttack(ti):
    to_attack = ti.xcom_pull(key='attack_SquareAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SquareAttack'
    else:
        return 'default'
def to_attack_SquareAttack(ti):
    to_attack = ti.xcom_pull(key='attack_SquareAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SquareAttack'
    else:
        return 'default'
def to_attack_SquareAttack(ti):
    to_attack = ti.xcom_pull(key='attack_SquareAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SquareAttack'
    else:
        return 'default'
def to_attack_TargetedUniversalPerturbation(ti):
    to_attack = ti.xcom_pull(key='attack_TargetedUniversalPerturbation',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_TargetedUniversalPerturbation'
    else:
        return 'default'
def to_attack_UniversalPerturbation(ti):
    to_attack = ti.xcom_pull(key='attack_UniversalPerturbation',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_UniversalPerturbation'
    else:
        return 'default'
def to_attack_VirtualAdversarialMethod(ti):
    to_attack = ti.xcom_pull(key='attack_VirtualAdversarialMethod',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_VirtualAdversarialMethod'
    else:
        return 'default'
def to_attack_Wasserstein(ti):
    to_attack = ti.xcom_pull(key='attack_Wasserstein',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_Wasserstein'
    else:
        return 'default'
def to_attack_ZooAttack(ti):
    to_attack = ti.xcom_pull(key='attack_ZooAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ZooAttack'
    else:
        return 'default'
def to_attack_FrameSaliencyAttack(ti):
    to_attack = ti.xcom_pull(key='attack_FrameSaliencyAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_FrameSaliencyAttack'
    else:
        return 'default'
def to_attack_GeoDA(ti):
    to_attack = ti.xcom_pull(key='attack_GeoDA',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_GeoDA'
    else:
        return 'default'
def to_attack_ElasticNet(ti):
    to_attack = ti.xcom_pull(key='attack_ElasticNet',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ElasticNet'
    else:
        return 'default'
def to_attack_CarliniL2Method(ti):
    to_attack = ti.xcom_pull(key='attack_CarliniL2Method',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_CarliniL2Method'
    else:
        return 'default'
def to_attack_BoundaryAttack(ti):
    to_attack = ti.xcom_pull(key='attack_BoundaryAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_BoundaryAttack'
    else:
        return 'default'
def to_attack_AutoProjectedGradientDescent(ti):
    to_attack = ti.xcom_pull(key='attack_AutoProjectedGradientDescent',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_AutoProjectedGradientDescent'
    else:
        return 'default'
def to_attack_DeepFool(ti):
    to_attack = ti.xcom_pull(key='attack_DeepFool',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_DeepFool'
    else:
        return 'default'
def to_attack_AutoAttack(ti):
    to_attack = ti.xcom_pull(key='attack_AutoAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_AutoAttack'
    else:
        return 'default'
def to_attack_LowProFool(ti):
    to_attack = ti.xcom_pull(key='attack_LowProFool',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_LowProFool'
    else:
        return 'default'
def to_attack_NewtonFool(ti):
    to_attack = ti.xcom_pull(key='attack_NewtonFool',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_NewtonFool'
    else:
        return 'default'
def to_attack_MalwareGDTensorFlow(ti):
    to_attack = ti.xcom_pull(key='attack_MalwareGDTensorFlow',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_MalwareGDTensorFlow'
    else:
        return 'default'
def to_attack_PixelAttack(ti):
    to_attack = ti.xcom_pull(key='attack_PixelAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_PixelAttack'
    else:
        return 'default'
def to_attack_SaliencyMapMethod(ti):
    to_attack = ti.xcom_pull(key='attack_SaliencyMapMethod',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SaliencyMapMethod'
    else:
        return 'default'
def to_attack_ShadowAttack(ti):
    to_attack = ti.xcom_pull(key='attack_ShadowAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ShadowAttack'
    else:
        return 'default'
def to_attack_SpatialTransformation(ti):
    to_attack = ti.xcom_pull(key='attack_SpatialTransformation',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SpatialTransformation'
    else:
        return 'default'
def to_attack_ShapeShifter(ti):
    to_attack = ti.xcom_pull(key='attack_ShapeShifter',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ShapeShifter'
    else:
        return 'default'
def to_attack_SignOPTAttack(ti):
    to_attack = ti.xcom_pull(key='attack_SignOPTAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SignOPTAttack'
    else:
        return 'default'
def to_attack_AdversarialPatch(ti):
    to_attack = ti.xcom_pull(key='attack_AdversarialPatch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_AdversarialPatch'
    else:
        return 'default'
def to_attack_AdversarialPatchPyTorch(ti):
    to_attack = ti.xcom_pull(key='attack_AdversarialPatchPyTorch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_AdversarialPatchPyTorch'
    else:
        return 'default'
def to_attack_FeatureAdversariesPyTorch(ti):
    to_attack = ti.xcom_pull(key='attack_FeatureAdversariesPyTorch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_FeatureAdversariesPyTorch'
    else:
        return 'default'
def to_attack_GRAPHITEBlackbox(ti):
    to_attack = ti.xcom_pull(key='attack_GRAPHITEBlackbox',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_GRAPHITEBlackbox'
    else:
        return 'default'
def to_attack_GRAPHITEWhiteboxPyTorch(ti):
    to_attack = ti.xcom_pull(key='attack_GRAPHITEWhiteboxPyTorch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_GRAPHITEWhiteboxPyTorch'
    else:
        return 'default'
def to_attack_LaserAttack(ti):
    to_attack = ti.xcom_pull(key='attack_LaserAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_LaserAttack'
    else:
        return 'default'
def to_attack_OverTheAirFlickeringPyTorch(ti):
    to_attack = ti.xcom_pull(key='attack_OverTheAirFlickeringPyTorch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_OverTheAirFlickeringPyTorch'
    else:
        return 'default'
def to_attack_BasicIterativeMethod(ti):
    to_attack = ti.xcom_pull(key='attack_BasicIterativeMethod',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_BasicIterativeMethod'
    else:
        return 'default'

def to_attack_FastGradientMethod(ti):
    to_attack = ti.xcom_pull(key='attack_FastGradientMethod',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_FastGradientMethod'
    else:
        return 'default'
def to_attack_ProjectedGradientDescent(ti):
    to_attack = ti.xcom_pull(key='attack_ProjectedGradientDescent',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ProjectedGradientDescent'
    else:
        return 'default'



def attack_auto_projected_gradient_descent(ti):
    model_acc, adversarial_examples  = attack(auto_projected_gradient_descent)
    ti.xcom_push(key='attack_auto_projected_gradient_descent_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_auto_projected_gradient_descent_score', value=model_acc)



def attack_SquareAttack(ti):
    model_acc, adversarial_examples  = attack(SquareAttack)
    ti.xcom_push(key='attack_SquareAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_SquareAttack_score', value=model_acc)

def attack_SquareAttack(ti):
    model_acc, adversarial_examples  = attack(SquareAttack)
    ti.xcom_push(key='attack_SquareAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_SquareAttack_score', value=model_acc)

def attack_SquareAttack(ti):
    model_acc, adversarial_examples  = attack(SquareAttack)
    ti.xcom_push(key='attack_SquareAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_SquareAttack_score', value=model_acc)

def attack_TargetedUniversalPerturbation(ti):
    model_acc, adversarial_examples  = attack(TargetedUniversalPerturbation)
    ti.xcom_push(key='attack_TargetedUniversalPerturbation_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_TargetedUniversalPerturbation_score', value=model_acc)

def attack_UniversalPerturbation(ti):
    model_acc, adversarial_examples  = attack(UniversalPerturbation)
    ti.xcom_push(key='attack_UniversalPerturbation_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_UniversalPerturbation_score', value=model_acc)

def attack_VirtualAdversarialMethod(ti):
    model_acc, adversarial_examples  = attack(VirtualAdversarialMethod)
    ti.xcom_push(key='attack_VirtualAdversarialMethod_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_VirtualAdversarialMethod_score', value=model_acc)

def attack_Wasserstein(ti):
    model_acc, adversarial_examples  = attack(Wasserstein)
    ti.xcom_push(key='attack_Wasserstein_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_Wasserstein_score', value=model_acc)

def attack_ZooAttack(ti):
    model_acc, adversarial_examples  = attack(ZooAttack)
    ti.xcom_push(key='attack_ZooAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_ZooAttack_score', value=model_acc)

def attack_FrameSaliencyAttack(ti):
    model_acc, adversarial_examples  = attack(FrameSaliencyAttack)
    ti.xcom_push(key='attack_FrameSaliencyAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_FrameSaliencyAttack_score', value=model_acc)

def attack_GeoDA(ti):
    model_acc, adversarial_examples  = attack(GeoDA)
    ti.xcom_push(key='attack_GeoDA_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_GeoDA_score', value=model_acc)

def attack_ElasticNet(ti):
    model_acc, adversarial_examples  = attack(ElasticNet)
    ti.xcom_push(key='attack_ElasticNet_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_ElasticNet_score', value=model_acc)

def attack_CarliniL2Method(ti):
    model_acc, adversarial_examples  = attack(CarliniL2Method)
    ti.xcom_push(key='attack_CarliniL2Method_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_CarliniL2Method_score', value=model_acc)

def attack_BoundaryAttack(ti):
    model_acc, adversarial_examples  = attack(BoundaryAttack)
    ti.xcom_push(key='attack_BoundaryAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_BoundaryAttack_score', value=model_acc)

def attack_AutoProjectedGradientDescent(ti):
    model_acc, adversarial_examples  = attack(AutoProjectedGradientDescent)
    ti.xcom_push(key='attack_AutoProjectedGradientDescent_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_AutoProjectedGradientDescent_score', value=model_acc)

def attack_DeepFool(ti):
    model_acc, adversarial_examples  = attack(DeepFool)
    ti.xcom_push(key='attack_DeepFool_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_DeepFool_score', value=model_acc)

def attack_AutoAttack(ti):
    model_acc, adversarial_examples  = attack(AutoAttack)
    ti.xcom_push(key='attack_AutoAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_AutoAttack_score', value=model_acc)

def attack_LowProFool(ti):
    model_acc, adversarial_examples  = attack(LowProFool)
    ti.xcom_push(key='attack_LowProFool_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_LowProFool_score', value=model_acc)

def attack_NewtonFool(ti):
    model_acc, adversarial_examples  = attack(NewtonFool)
    ti.xcom_push(key='attack_NewtonFool_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_NewtonFool_score', value=model_acc)

def attack_MalwareGDTensorFlow(ti):
    model_acc, adversarial_examples  = attack(MalwareGDTensorFlow)
    ti.xcom_push(key='attack_MalwareGDTensorFlow_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_MalwareGDTensorFlow_score', value=model_acc)

def attack_PixelAttack(ti):
    model_acc, adversarial_examples  = attack(PixelAttack)
    ti.xcom_push(key='attack_PixelAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_PixelAttack_score', value=model_acc)

def attack_SaliencyMapMethod(ti):
    model_acc, adversarial_examples  = attack(SaliencyMapMethod)
    ti.xcom_push(key='attack_SaliencyMapMethod_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_SaliencyMapMethod_score', value=model_acc)

def attack_ShadowAttack(ti):
    model_acc, adversarial_examples  = attack(ShadowAttack)
    ti.xcom_push(key='attack_ShadowAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_ShadowAttack_score', value=model_acc)

def attack_SpatialTransformation(ti):
    model_acc, adversarial_examples  = attack(SpatialTransformation)
    ti.xcom_push(key='attack_SpatialTransformation_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_SpatialTransformation_score', value=model_acc)

def attack_ShapeShifter(ti):
    model_acc, adversarial_examples  = attack(ShapeShifter)
    ti.xcom_push(key='attack_ShapeShifter_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_ShapeShifter_score', value=model_acc)

def attack_SignOPTAttack(ti):
    model_acc, adversarial_examples  = attack(SignOPTAttack)
    ti.xcom_push(key='attack_SignOPTAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_SignOPTAttack_score', value=model_acc)

def attack_AdversarialPatch(ti):
    model_acc, adversarial_examples  = attack(AdversarialPatch)
    ti.xcom_push(key='attack_AdversarialPatch_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_AdversarialPatch_score', value=model_acc)

def attack_AdversarialPatchPyTorch(ti):
    model_acc, adversarial_examples  = attack(AdversarialPatchPyTorch)
    ti.xcom_push(key='attack_AdversarialPatchPyTorch_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_AdversarialPatchPyTorch_score', value=model_acc)

def attack_FeatureAdversariesPyTorch(ti):
    model_acc, adversarial_examples  = attack(FeatureAdversariesPyTorch)
    ti.xcom_push(key='attack_FeatureAdversariesPyTorch_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_FeatureAdversariesPyTorch_score', value=model_acc)

def attack_GRAPHITEBlackbox(ti):
    model_acc, adversarial_examples  = attack(GRAPHITEBlackbox)
    ti.xcom_push(key='attack_GRAPHITEBlackbox_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_GRAPHITEBlackbox_score', value=model_acc)

def attack_GRAPHITEWhiteboxPyTorch(ti):
    model_acc, adversarial_examples  = attack(GRAPHITEWhiteboxPyTorch)
    ti.xcom_push(key='attack_GRAPHITEWhiteboxPyTorch_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_GRAPHITEWhiteboxPyTorch_score', value=model_acc)

def attack_LaserAttack(ti):
    model_acc, adversarial_examples  = attack(LaserAttack)
    ti.xcom_push(key='attack_LaserAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_LaserAttack_score', value=model_acc)

def attack_OverTheAirFlickeringPyTorch(ti):
    model_acc, adversarial_examples  = attack(OverTheAirFlickeringPyTorch)
    ti.xcom_push(key='attack_OverTheAirFlickeringPyTorch_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_OverTheAirFlickeringPyTorch_score', value=model_acc)

def attack_BasicIterativeMethod(ti):
    model_acc, adversarial_examples  = attack(BasicIterativeMethod)
    ti.xcom_push(key='attack_BasicIterativeMethod_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_BasicIterativeMethod_score', value=model_acc)

def attack_FastGradientMethod(ti):
    model_acc, adversarial_examples = attack(FastGradientMethod)
    ti.xcom_push(key='attack_FastGradientMethod_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_FastGradientMethod_score', value=model_acc)

def attack_ProjectedGradientDescent(ti):
    model_acc,adversarial_examples = attack(ProjectedGradientDescent)
    ti.xcom_push(key='attack_ProjectedGradientDescent_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_ProjectedGradientDescent_score', value=model_acc)

def attack(attack_obj):
    data = get_data()
    estimator = get_estimator()
    wrap_attack = attack_obj(estimator=estimator, eps=0.2)
    x_train, x_test = next(data)
    adversarial_examples = wrap_attack.generate(np.asarray(x_train))
    prediction_softmax_results = estimator.predict(adversarial_examples)
    prediction_results = np.argmax(prediction_softmax_results, axis=1)
    y_train, y_test = next(data)
    model_acc = accuracy_score(y_test, prediction_results)
    sent_model_after_attack(estimator)
    return model_acc, adversarial_examples
def default(ti):
    ti.xcom_push(key='Default', value=0)

def choose_best_attack(ti):

    attacks_scores = {'attack_BasicIterativeMethod':0,'attack_FastGradientMethod':0,'attack_ProjectedGradientDescent':0}
    for index, attacks_score in enumerate(attacks_scores):  # attack score is the key
        score = ti.xcom_pull(key=attacks_score + "_score",
                             task_ids=f'{attacks_score}')
        if score:
            attacks_scores[attacks_score] = score


    best_score = max(attacks_scores.values())
    metadata = load_from_bucket(file_name='attack_defence_metadata.json',as_json=True)
    for key, val in attacks_scores.items():
        if val == best_score:
            metadata['attack_best_scores'].append((key,val))
            ti.xcom_push(key=f'best: {key} in round {metadata["cycles"]} : ', value=val)
            adv_examples = ti.xcom_pull(key=key + "_adv",
                         task_ids=f'{key}')
            upload_to_bucket(obj=adv_examples,file_name="adv.csv", as_csv=True)
            upload_to_bucket(obj=metadata,file_name='attack_defence_metadata.json',as_json=True)
            return
def set_or_create(ti):
    try:
        metadata = load_from_bucket(file_name='attack_defence_metadata.json',as_json=True)
        ti.xcom_push(key='metadata', value='exist')
    except:
        ti.xcom_push(key='metadata', value='not exist...creating....')
        metadata = {"cycles": 0, "attack_best_scores": [], "defence_best_scores": []}
        upload_to_bucket(obj=metadata, file_name='attack_defence_metadata.json',as_json=True)
        ti.xcom_push(key='metadata', value='uploaded')


with DAG('multi_attack_dag', schedule_interval='@daily', default_args=default_args, catchup=False,dagrun_timeout=timedelta(minutes=240)) as dag:

    choose_attack = PythonOperator(
            task_id=f'choose_attack',
            python_callable=pars_json
        )

    metadata = PythonOperator(
        task_id="metadata",
        python_callable=set_or_create
    )

    branch_SquareAttack = BranchPythonOperator(
        task_id='to_attack_SquareAttack',
        python_callable=to_attack_SquareAttack
    )

    run_attack_SquareAttack = PythonOperator(
            task_id=f"attack_SquareAttack",
            python_callable=attack_SquareAttack)
    branch_TargetedUniversalPerturbation = BranchPythonOperator(
        task_id='to_attack_TargetedUniversalPerturbation',
        python_callable=to_attack_TargetedUniversalPerturbation
    )

    run_attack_TargetedUniversalPerturbation = PythonOperator(
            task_id=f"attack_TargetedUniversalPerturbation",
            python_callable=attack_TargetedUniversalPerturbation)
    branch_UniversalPerturbation = BranchPythonOperator(
        task_id='to_attack_UniversalPerturbation',
        python_callable=to_attack_UniversalPerturbation
    )

    run_attack_UniversalPerturbation = PythonOperator(
            task_id=f"attack_UniversalPerturbation",
            python_callable=attack_UniversalPerturbation)
    branch_VirtualAdversarialMethod = BranchPythonOperator(
        task_id='to_attack_VirtualAdversarialMethod',
        python_callable=to_attack_VirtualAdversarialMethod
    )

    run_attack_VirtualAdversarialMethod = PythonOperator(
            task_id=f"attack_VirtualAdversarialMethod",
            python_callable=attack_VirtualAdversarialMethod)
    branch_Wasserstein = BranchPythonOperator(
        task_id='to_attack_Wasserstein',
        python_callable=to_attack_Wasserstein
    )

    run_attack_Wasserstein = PythonOperator(
            task_id=f"attack_Wasserstein",
            python_callable=attack_Wasserstein)
    branch_ZooAttack = BranchPythonOperator(
        task_id='to_attack_ZooAttack',
        python_callable=to_attack_ZooAttack
    )

    run_attack_ZooAttack = PythonOperator(
            task_id=f"attack_ZooAttack",
            python_callable=attack_ZooAttack)
    branch_FrameSaliencyAttack = BranchPythonOperator(
        task_id='to_attack_FrameSaliencyAttack',
        python_callable=to_attack_FrameSaliencyAttack
    )

    run_attack_FrameSaliencyAttack = PythonOperator(
            task_id=f"attack_FrameSaliencyAttack",
            python_callable=attack_FrameSaliencyAttack)
    branch_GeoDA = BranchPythonOperator(
        task_id='to_attack_GeoDA',
        python_callable=to_attack_GeoDA
    )

    run_attack_GeoDA = PythonOperator(
            task_id=f"attack_GeoDA",
            python_callable=attack_GeoDA)
    branch_ElasticNet = BranchPythonOperator(
        task_id='to_attack_ElasticNet',
        python_callable=to_attack_ElasticNet
    )

    run_attack_ElasticNet = PythonOperator(
            task_id=f"attack_ElasticNet",
            python_callable=attack_ElasticNet)
    branch_CarliniL2Method = BranchPythonOperator(
        task_id='to_attack_CarliniL2Method',
        python_callable=to_attack_CarliniL2Method
    )

    run_attack_CarliniL2Method = PythonOperator(
            task_id=f"attack_CarliniL2Method",
            python_callable=attack_CarliniL2Method)
    branch_BoundaryAttack = BranchPythonOperator(
        task_id='to_attack_BoundaryAttack',
        python_callable=to_attack_BoundaryAttack
    )

    run_attack_BoundaryAttack = PythonOperator(
            task_id=f"attack_BoundaryAttack",
            python_callable=attack_BoundaryAttack)
    branch_AutoProjectedGradientDescent = BranchPythonOperator(
        task_id='to_attack_AutoProjectedGradientDescent',
        python_callable=to_attack_AutoProjectedGradientDescent
    )

    run_attack_AutoProjectedGradientDescent = PythonOperator(
            task_id=f"attack_AutoProjectedGradientDescent",
            python_callable=attack_AutoProjectedGradientDescent)
    branch_DeepFool = BranchPythonOperator(
        task_id='to_attack_DeepFool',
        python_callable=to_attack_DeepFool
    )

    run_attack_DeepFool = PythonOperator(
            task_id=f"attack_DeepFool",
            python_callable=attack_DeepFool)
    branch_AutoAttack = BranchPythonOperator(
        task_id='to_attack_AutoAttack',
        python_callable=to_attack_AutoAttack
    )

    run_attack_AutoAttack = PythonOperator(
            task_id=f"attack_AutoAttack",
            python_callable=attack_AutoAttack)
    branch_LowProFool = BranchPythonOperator(
        task_id='to_attack_LowProFool',
        python_callable=to_attack_LowProFool
    )

    run_attack_LowProFool = PythonOperator(
            task_id=f"attack_LowProFool",
            python_callable=attack_LowProFool)
    branch_NewtonFool = BranchPythonOperator(
        task_id='to_attack_NewtonFool',
        python_callable=to_attack_NewtonFool
    )

    run_attack_NewtonFool = PythonOperator(
            task_id=f"attack_NewtonFool",
            python_callable=attack_NewtonFool)
    branch_MalwareGDTensorFlow = BranchPythonOperator(
        task_id='to_attack_MalwareGDTensorFlow',
        python_callable=to_attack_MalwareGDTensorFlow
    )

    run_attack_MalwareGDTensorFlow = PythonOperator(
            task_id=f"attack_MalwareGDTensorFlow",
            python_callable=attack_MalwareGDTensorFlow)
    branch_PixelAttack = BranchPythonOperator(
        task_id='to_attack_PixelAttack',
        python_callable=to_attack_PixelAttack
    )

    run_attack_PixelAttack = PythonOperator(
            task_id=f"attack_PixelAttack",
            python_callable=attack_PixelAttack)
    branch_SaliencyMapMethod = BranchPythonOperator(
        task_id='to_attack_SaliencyMapMethod',
        python_callable=to_attack_SaliencyMapMethod
    )

    run_attack_SaliencyMapMethod = PythonOperator(
            task_id=f"attack_SaliencyMapMethod",
            python_callable=attack_SaliencyMapMethod)
    branch_ShadowAttack = BranchPythonOperator(
        task_id='to_attack_ShadowAttack',
        python_callable=to_attack_ShadowAttack
    )

    run_attack_ShadowAttack = PythonOperator(
            task_id=f"attack_ShadowAttack",
            python_callable=attack_ShadowAttack)
    branch_SpatialTransformation = BranchPythonOperator(
        task_id='to_attack_SpatialTransformation',
        python_callable=to_attack_SpatialTransformation
    )

    run_attack_SpatialTransformation = PythonOperator(
            task_id=f"attack_SpatialTransformation",
            python_callable=attack_SpatialTransformation)
    branch_ShapeShifter = BranchPythonOperator(
        task_id='to_attack_ShapeShifter',
        python_callable=to_attack_ShapeShifter
    )

    run_attack_ShapeShifter = PythonOperator(
            task_id=f"attack_ShapeShifter",
            python_callable=attack_ShapeShifter)
    branch_SignOPTAttack = BranchPythonOperator(
        task_id='to_attack_SignOPTAttack',
        python_callable=to_attack_SignOPTAttack
    )

    run_attack_SignOPTAttack = PythonOperator(
            task_id=f"attack_SignOPTAttack",
            python_callable=attack_SignOPTAttack)
    branch_AdversarialPatch = BranchPythonOperator(
        task_id='to_attack_AdversarialPatch',
        python_callable=to_attack_AdversarialPatch
    )

    run_attack_AdversarialPatch = PythonOperator(
            task_id=f"attack_AdversarialPatch",
            python_callable=attack_AdversarialPatch)
    branch_AdversarialPatchPyTorch = BranchPythonOperator(
        task_id='to_attack_AdversarialPatchPyTorch',
        python_callable=to_attack_AdversarialPatchPyTorch
    )

    run_attack_AdversarialPatchPyTorch = PythonOperator(
            task_id=f"attack_AdversarialPatchPyTorch",
            python_callable=attack_AdversarialPatchPyTorch)
    branch_FeatureAdversariesPyTorch = BranchPythonOperator(
        task_id='to_attack_FeatureAdversariesPyTorch',
        python_callable=to_attack_FeatureAdversariesPyTorch
    )

    run_attack_FeatureAdversariesPyTorch = PythonOperator(
            task_id=f"attack_FeatureAdversariesPyTorch",
            python_callable=attack_FeatureAdversariesPyTorch)
    branch_GRAPHITEBlackbox = BranchPythonOperator(
        task_id='to_attack_GRAPHITEBlackbox',
        python_callable=to_attack_GRAPHITEBlackbox
    )

    run_attack_GRAPHITEBlackbox = PythonOperator(
            task_id=f"attack_GRAPHITEBlackbox",
            python_callable=attack_GRAPHITEBlackbox)
    branch_GRAPHITEWhiteboxPyTorch = BranchPythonOperator(
        task_id='to_attack_GRAPHITEWhiteboxPyTorch',
        python_callable=to_attack_GRAPHITEWhiteboxPyTorch
    )

    run_attack_GRAPHITEWhiteboxPyTorch = PythonOperator(
            task_id=f"attack_GRAPHITEWhiteboxPyTorch",
            python_callable=attack_GRAPHITEWhiteboxPyTorch)
    branch_LaserAttack = BranchPythonOperator(
        task_id='to_attack_LaserAttack',
        python_callable=to_attack_LaserAttack
    )

    run_attack_LaserAttack = PythonOperator(
            task_id=f"attack_LaserAttack",
            python_callable=attack_LaserAttack)
    branch_OverTheAirFlickeringPyTorch = BranchPythonOperator(
        task_id='to_attack_OverTheAirFlickeringPyTorch',
        python_callable=to_attack_OverTheAirFlickeringPyTorch
    )

    run_attack_OverTheAirFlickeringPyTorch = PythonOperator(
            task_id=f"attack_OverTheAirFlickeringPyTorch",
            python_callable=attack_OverTheAirFlickeringPyTorch)
    branch_auto_projected_gradient_descent = BranchPythonOperator(
        task_id='to_attack_auto_projected_gradient_descent',
        python_callable=to_attack_auto_projected_gradient_descent
    )

    run_attack_auto_projected_gradient_descent = PythonOperator(
            task_id=f"attack_auto_projected_gradient_descent",
            python_callable=attack_auto_projected_gradient_descent)

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
            task_id=f"default",
            python_callable=default
        )
    choose_best = PythonOperator(
            task_id=f"choose_best",
            python_callable=choose_best_attack,
            trigger_rule='none_failed'
        )
    trigger_defence = TriggerDagRunOperator(
        task_id='trigger_defence',
        trigger_dag_id='multi_defence_dag'
    )
    #
    choose_attack >> metadata >> branch_BasicIterativeMethod >> [run_attack_BasicIterativeMethod, run_default] >> choose_best
    choose_attack >> metadata >> branch_FastGradientMethod >> [run_attack_FastGradientMethod, run_default] >> choose_best
    choose_attack >> metadata >> branch_ProjectedGradientDescent >> [run_attack_ProjectedGradientDescent, run_default] >> choose_best
    choose_attack >> metadata >> branch_auto_projected_gradient_descent >> [run_attack_auto_projected_gradient_descent, run_default] >> choose_best
    choose_attack >> metadata >> branch_SquareAttack >> [run_attack_SquareAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_TargetedUniversalPerturbation >> [run_attack_TargetedUniversalPerturbation, run_default] >> choose_best
    choose_attack >> metadata >> branch_UniversalPerturbation >> [run_attack_UniversalPerturbation, run_default] >> choose_best
    choose_attack >> metadata >> branch_VirtualAdversarialMethod >> [run_attack_VirtualAdversarialMethod, run_default] >> choose_best
    choose_attack >> metadata >> branch_Wasserstein >> [run_attack_Wasserstein, run_default] >> choose_best
    choose_attack >> metadata >> branch_ZooAttack >> [run_attack_ZooAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_FrameSaliencyAttack >> [run_attack_FrameSaliencyAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_GeoDA >> [run_attack_GeoDA, run_default] >> choose_best
    choose_attack >> metadata >> branch_ElasticNet >> [run_attack_ElasticNet, run_default] >> choose_best
    choose_attack >> metadata >> branch_CarliniL2Method >> [run_attack_CarliniL2Method, run_default] >> choose_best
    choose_attack >> metadata >> branch_BoundaryAttack >> [run_attack_BoundaryAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_AutoProjectedGradientDescent >> [run_attack_AutoProjectedGradientDescent, run_default] >> choose_best
    choose_attack >> metadata >> branch_DeepFool >> [run_attack_DeepFool, run_default] >> choose_best
    choose_attack >> metadata >> branch_AutoAttack >> [run_attack_AutoAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_LowProFool >> [run_attack_LowProFool, run_default] >> choose_best
    choose_attack >> metadata >> branch_NewtonFool >> [run_attack_NewtonFool, run_default] >> choose_best
    choose_attack >> metadata >> branch_MalwareGDTensorFlow >> [run_attack_MalwareGDTensorFlow, run_default] >> choose_best
    choose_attack >> metadata >> branch_PixelAttack >> [run_attack_PixelAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_SaliencyMapMethod >> [run_attack_SaliencyMapMethod, run_default] >> choose_best
    choose_attack >> metadata >> branch_ShadowAttack >> [run_attack_ShadowAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_SpatialTransformation >> [run_attack_SpatialTransformation, run_default] >> choose_best
    choose_attack >> metadata >> branch_ShapeShifter >> [run_attack_ShapeShifter, run_default] >> choose_best
    choose_attack >> metadata >> branch_SignOPTAttack >> [run_attack_SignOPTAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_AdversarialPatch >> [run_attack_AdversarialPatch, run_default] >> choose_best
    choose_attack >> metadata >> branch_AdversarialPatchPyTorch >> [run_attack_AdversarialPatchPyTorch, run_default] >> choose_best
    choose_attack >> metadata >> branch_FeatureAdversariesPyTorch >> [run_attack_FeatureAdversariesPyTorch, run_default] >> choose_best
    choose_attack >> metadata >> branch_GRAPHITEBlackbox >> [run_attack_GRAPHITEBlackbox, run_default] >> choose_best
    choose_attack >> metadata >> branch_GRAPHITEWhiteboxPyTorch >> [run_attack_GRAPHITEWhiteboxPyTorch, run_default] >> choose_best
    choose_attack >> metadata >> branch_LaserAttack >> [run_attack_LaserAttack, run_default] >> choose_best
    choose_attack >> metadata >> branch_OverTheAirFlickeringPyTorch >> [run_attack_OverTheAirFlickeringPyTorch, run_default] >> choose_best
    choose_best >> trigger_defence