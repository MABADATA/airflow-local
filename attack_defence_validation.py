from art.attacks.evasion import CarliniL2Method, BasicIterativeMethod,FastGradientMethod,ProjectedGradientDescent,\
    MomentumIterativeMethod,NewtonFool,ProjectedGradientDescentPyTorch,UniversalPerturbation
from art.defences.postprocessor import GaussianNoise,ReverseSigmoid,Rounded ,HighConfidence
from skopt.space import Integer
from Bucket_loader import Bucket_loader
class AttackDefenceValidator:
    def __init__(self):
        self.HyperParameters_ranges = {'CarliniL2Method': {'confidence': 0.2, "obj": CarliniL2Method},
                          'BasicIterativeMethod': {'eps': 0.3, 'eps_step': 0.15, 'max_iter': 10000,"obj": BasicIterativeMethod},
                          'FastGradientMethod': {'eps': 0.005,"obj": FastGradientMethod},
                          'ProjectedGradientDescent': {'eps': 0.03, 'max_iter': 10000,"obj":ProjectedGradientDescent},
                          'MomentumIterativeMethod': {'eps': 0.1,"obj":MomentumIterativeMethod},
                          'NewtonFool': {'eta': 0.035,"obj":NewtonFool},
                          'ProjectedGradientDescentPyTorch': {'eps': 0.2, 'random_eps': True,"obj": ProjectedGradientDescentPyTorch
                                                              },
                          'UniversalPerturbation': {'attacker': 'deepfool', 'delta': 0.5,"obj": UniversalPerturbation}}

        self.__file_loader = Bucket_loader()
    def validate(self):
        """
        Validates if the actual defences that the user wants can run by checking the parameters.
        We get metadata of the attack and defences in the following structure:
        {attack: {'CarliniL2Method': bool,
                      'BasicIterativeMethod': bool,
                      'FastGradientMethod': bool,
                      'ProjectedGradientDescent': bool,
                      'MomentumIterativeMethod': bool,
                      'NewtonFool': bool,
                      'ProjectedGradientDescentPyTorch':bool,
                      'UniversalPerturbation': bool
                      },
        Where the bool value is True it means that the user want this defence, and where it's False other wise.
        :return: None. The method does not return, but only changes the metadata if an attack or defence is not valid.
        """
        attack_defence_metadata = self.__file_loader.get_attack_defence_json()
        attack_metadata = attack_defence_metadata['attack']
        estimator = self.__file_loader.get_estimator()

        for attack, bool_val in attack_metadata.items():
            if bool_val:
                attack_obj = self.HyperParameters_ranges[attack]["obj"]
                attack_estimator_req = attack_obj._estimator_requirements
                if not issubclass(estimator,attack_estimator_req):
                    attack_metadata[attack] = False

        attack_defence_metadata['attack'] = attack_metadata
        self.__file_loader.upload(attack_defence_metadata,"attack_defence_metadata", to_pickle=False)








