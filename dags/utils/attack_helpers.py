from sklearn.metrics import accuracy_score
from skopt import gp_minimize
from .helpers import *


def optimize_evasion_attack(attack,classifier,data,true_labels):
    """
    This function only responsible for optimize the hyper parameters a post processor type defense.
    :param attack:
    :return: the optimized defense instance.
    """
    print(f'I am right here on {attack}')
    attack_parameter_range = {'CarliniL2Method': {'confidence': (0, 0.5)},
                              'BasicIterativeMethod': {'eps': (0.0001, 1), 'eps_step': (0.1, 1)}
        , 'FastGradientMethod': {'eps': (0.0001, 1)},
                              'ProjectedGradientDescent': {'eps': (0.0001, 1)},
                              'MomentumIterativeMethod': {'eps': (0.0001, 1)},
                              'NewtonFool': {'eta': (0.01, 0.1)},
                              'ProjectedGradientDescent': {'eps': (0.001, 1), 'random_eps': (True, False)}
        , 'ProjectedGradientDescentPyTorch': {'eps': (0.001, 1), 'random_eps': (True, False)},
                              'UniversalPerturbation': {
                                  'attacker': ('carlini', 'carlini_inf', 'deepfool', 'fgsm', 'bim'),
                                  'delta': (0.1, 0.5)}
                              }
    attacks_parameters = {'CarliniL2Method': ['confidence'], 'BasicIterativeMethod': ['eps', 'eps_step'],
                          'FastGradientMethod': ['eps'], 'ProjectedGradientDescent': ['eps'],
                          'MomentumIterativeMethod': ['eps'], 'MomentumIterativeMethod': ['eps']
        , 'NewtonFool': ['eta'], 'ProjectedGradientDescent': ['eps', 'random_eps'],
                          'ProjectedGradientDescentPyTorch': ['eps', 'random_eps'],
                          'UniversalPerturbation': ['attacker', 'delta']}
    def _optimize(attack_to_optimize, hyperparams, classifier,data,true_labels):

        HP = {k: v for k, v in zip(attacks_parameters[attack.__name__], hyperparams)}
        print(f'This is the attack : {attack} and its HP : {HP}')
        attack_to_optimize = globals()[attack.__name__](classifier, **HP)
        # adv_examples = attack_to_optimize(self.classifier).generate(self.data)
        adv_examples = attack_to_optimize.generate(data)
        prediction_softmax_results_def = classifier.predict(adv_examples)
        prediction_results_def = np.argmax(prediction_softmax_results_def, axis=1)
        # get the model's accuracy after the denfese applied and return
        acc = accuracy_score(true_labels, prediction_results_def)
        return -acc

    search_space = [v for k, v in attack_parameter_range[attack.__name__].items()]
    func = lambda params: _optimize(attack, params,classifier=classifier,data=data,true_labels=true_labels)
    print(func)
    result = gp_minimize(func, search_space, n_calls=10)

    final_HP = {k: v for k, v in zip(attacks_parameters[attack.__name__], result.x)}
    optimized_attack = globals()[attack.__name__](classifier,**final_HP)
    print(f'This is the optimized attack {optimized_attack}')
    return optimized_attack


def pars_attack_json(ti):
    attack_json = json.loads(load_from_bucket('attack_defence_metadata.json'))['attack']

    ti.xcom_push(key='Module_pars', value=__name__)
    for attack, bool_val in attack_json.items():
        ti.xcom_push(key=attack, value=bool_val)

def set_or_create(ti):
    try:
        metadata = load_from_bucket(file_name='attack_defence_dag_metadata.json',as_json=True)
        ti.xcom_push(key='metadata', value='exist')
    except:
        ti.xcom_push(key='metadata', value='not exist...creating....')
        metadata = {"cycles": 0, "attack_best_scores": [], "defence_best_scores": []}
        upload_to_bucket(obj=metadata, file_name='attack_defence_dag_metadata.json',as_json=True)
        ti.xcom_push(key='metadata', value='uploaded')

def sent_model_after_attack(estimator):
    upload_to_bucket(estimator.model, 'ML_model.pickle')

def set_attack_params(attack, params,estimator ):
    attack_with_params = attack(**params,estimator=estimator)
    return attack_with_params

def choose_best_attack(ti):

    attack_score_pair = {'attack_BasicIterativeMethod': 0,
                   'attack_FastGradientMethod': 0, 'attack_ProjectedGradientDescent': 0,
                   'SquareAttack':0,
                  'TargetedUniversalPerturbation':0,
                  'UniversalPerturbation':0,
                  'VirtualAdversarialMethod':0, 'Wasserstein':0, 'ZooAttack':0,
                  'FrameSaliencyAttack':0, 'GeoDA':0, 'ElasticNet':0, 'CarliniL2Method':0, 'BoundaryAttack':0,
                  'AutoProjectedGradientDescent':0, 'DeepFool':0, 'AutoAttack':0,
                   'LowProFool':0, 'NewtonFool':0,
                  'MalwareGDTensorFlow':0, 'PixelAttack':0, 'SaliencyMapMethod':0, 'ShadowAttack':0,
                  'SpatialTransformation':0, 'ShapeShifter':0, 'SignOPTAttack':0, 'AdversarialPatch':True, 'AdversarialPatchPyTorch':True,
                  'FeatureAdversariesPyTorch':0, 'GRAPHITEBlackbox':0, 'GRAPHITEWhiteboxPyTorch':0, 'LaserAttack':0,
                  'OverTheAirFlickeringPyTorch':0
    }
    for index, attack_score in enumerate(attack_score_pair):  # attack score is the key
        score = ti.xcom_pull(key=attack_score + "_score",
                             task_ids=f'{attack_score}')
        if score:
            attack_score_pair[attack_score] = score

    best_attack = max(attack_score_pair, key=attack_score_pair.get)
    best_score = attack_score_pair[best_attack]
    metadata = load_from_bucket(file_name='attack_defence_dag_metadata.json',as_json=True)
    metadata['attack_best_scores'].append((best_attack,best_score))
    ti.xcom_push(key=f'best: {best_attack} in round {metadata["cycles"]} : ', value=best_score)
    adv_examples = ti.xcom_pull(key=best_attack + "_adv",
                 task_ids=f'{best_attack}')
    upload_to_bucket(obj=adv_examples,file_name="adv.csv", as_csv=True)
    upload_to_bucket(obj=metadata,file_name='attack_defence_dag_metadata.json',as_json=True)
    return
