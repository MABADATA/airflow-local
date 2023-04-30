from dags.utils.helpers import *
import json
from torch.optim import SGD
from skopt import gp_minimize
from sklearn.metrics import accuracy_score


def ser(obj):
    try:
        ser_obj = json.dumps(obj)
        return ser_obj
    except:
        return

def assign_vars(cls, args_dict,ML_model):
    if args_dict.get("optimizer"):
        optimizer = SGD(ML_model.parameters(), lr=0.01)
        args_dict["optimizer"] = optimizer
    if args_dict.get("loss"):
        loss = load_from_bucket('loss.pickle')
        args_dict["loss"] = loss
    obj = cls(**args_dict,model=ML_model)
    return obj
global attacks_parameters
global attack_parameter_range
attacks_parameters  = {'CarliniL2Method':['confidence'],'BasicIterativeMethod':['eps','eps_step'],
                                   'FastGradientMethod':['eps'],'ProjectedGradientDescent':['eps'],
                                   'MomentumIterativeMethod':['eps'],'MomentumIterativeMethod':['eps']
                                   ,'NewtonFool':['eta'],'ProjectedGradientDescent':['eps','random_eps'],
                                   'ProjectedGradientDescentPyTorch':['eps','random_eps'],'UniversalPerturbation':['attacker','delta']}
attack_parameter_range = {'CarliniL2Method': {'confidence': (0, 0.5)},
                                                'BasicIterativeMethod': {'eps': (0.0001, 1), 'eps_step': (0.1, 1)},
                                                'FastGradientMethod': {'eps': (0.0001, 1)},
                                                'ProjectedGradientDescent': {'eps': (0.0001, 1)},
                                                'MomentumIterativeMethod': {'eps': (0.0001, 1)},
                                                'NewtonFool': {'eta': (0.01, 0.1)},
                                                'ProjectedGradientDescent': {'eps': (0.001, 1), 'random_eps': (True, False)},
                                                'ProjectedGradientDescentPyTorch': {'eps': (0.001, 1), 'random_eps': (True, False)},
                                                'UniversalPerturbation': {
                                                'attacker': ('carlini', 'carlini_inf', 'deepfool', 'fgsm', 'bim'),
                                                'delta': (0.1, 0.5)}
                                       }
attack_parameter_range = {'CarliniL2Method': {'confidence': 0.2 },
                            'BasicIterativeMethod': {'eps': 0.3, 'eps_step': 0.15},
                            'FastGradientMethod': {'eps': 0.005},
                            'ProjectedGradientDescent': {'eps': 0.03},
                            'MomentumIterativeMethod': {'eps': 0.1},
                            'NewtonFool': {'eta': 0.035},
                            'ProjectedGradientDescent': {'eps': 0.001, 'random_eps': False},
                            'ProjectedGradientDescentPyTorch': {'eps': 0.2, 'random_eps': True},
                            'UniversalPerturbation': {'attacker':  'deepfool', 'delta':  0.5}
                   }



def set_attack_params(attack, params):
    attack_with_params = attack(**params)
    return attack_with_params


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

        HP = {k: v for k, v in zip(attacks_parameters[type(attack).__name__], hyperparams)}
        print(f'This is the attack : {attack} and its HP : {HP}')
        attack_to_optimize = globals()[type(attack).__name__](classifier, **HP)
        # adv_examples = attack_to_optimize(self.classifier).generate(self.data)
        adv_examples = attack_to_optimize.generate(data)
        prediction_softmax_results_def = classifier.predict(adv_examples)
        prediction_results_def = np.argmax(prediction_softmax_results_def, axis=1)
        # get the model's accuracy after the denfese applied and return
        acc = accuracy_score(true_labels, prediction_results_def)
        return -acc

    search_space = [v for k, v in attack_parameter_range[type(attack).__name__].items()]
    func = lambda params: _optimize(attack, params,classifier=classifier,data=data,true_labels=true_labels)
    print(func)
    result = gp_minimize(func, search_space, n_calls=10)

    final_HP = {k: v for k, v in zip(attacks_parameters[type(attack).__name__], result.x)}
    optimized_attack = globals()[type(attack).__name__](classifier,**final_HP)
    print(f'This is the optimized attack {optimized_attack}')
    return optimized_attack



if __name__ == '__main__':

    attacks = ['SquareAttack',
    'TargetedUniversalPerturbation',
    'UniversalPerturbation',
    'VirtualAdversarialMethod','Wasserstein','ZooAttack',
    'FrameSaliencyAttack','GeoDA','ElasticNet','CarliniL2Method','BoundaryAttack',
    'AutoProjectedGradientDescent','DeepFool','AutoAttack','LowProFool','NewtonFool',
    'MalwareGDTensorFlow','PixelAttack','SaliencyMapMethod','ShadowAttack',
    'SpatialTransformation','ShapeShifter','SignOPTAttack','AdversarialPatch','AdversarialPatchPyTorch',
    'FeatureAdversariesPyTorch','GRAPHITEBlackbox','GRAPHITEWhiteboxPyTorch','LaserAttack',
    'OverTheAirFlickeringPyTorch']
    # for attack in attacks:
    #     add_attack(attack)
    with open("Estimator_params.json", 'rb') as f:
        fil = pickle.load(f)
        print(fil)

