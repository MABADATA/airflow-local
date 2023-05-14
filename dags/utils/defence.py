import numpy as np
from helpers import get_data
from estimator_helpers import get_estimator
from helpers import load_from_bucket
from defence_helpers import *
from art.defences.postprocessor import GaussianNoise, ReverseSigmoid, Rounded,HighConfidence

def defend_GaussianNoise(ti):
    defence_accuracy,defense_HP = defence(GaussianNoise)
    defence_dict = {"accuracy": defence_accuracy, "HP": defense_HP}
    ti.xcom_push(key='defence_GaussianNoise', value=defence_dict)

def defend_ReverseSigmoid(ti):
    defence_accuracy,defense_HP = defence(ReverseSigmoid)
    defence_dict = {"accuracy": defence_accuracy, "HP": defense_HP}
    ti.xcom_push(key='defence_ReverseSigmoid', value=defence_dict)

def defend_Rounded(ti):
    defence_accuracy, defense_HP = defence(Rounded)
    defence_dict = {"accuracy": defence_accuracy, "HP": defense_HP}
    ti.xcom_push(key='defence_Rounded', value=defence_dict)
def defend_HighConfidence(ti):
    defence_accuracy, defense_HP = defence(HighConfidence)
    defence_dict = {"accuracy": defence_accuracy, "HP": defense_HP}
    ti.xcom_push(key='defence_HighConfidence', value=defence_dict)



def defence(post_defense):
    (x_train, y_train) ,(x_test, y_test) = get_data()
    estimator = get_estimator()
    estimator.fit(x_train ,y_train)
    adv_examples = load_from_bucket("adv.csv" ,as_csv=True)
    acc, opt_defense_HP = try_defense(classifier=estimator,
                                   defense=post_defense,
                                   adv_examples=adv_examples,
                                   true_labels=y_test)
    return acc, opt_defense_HP


