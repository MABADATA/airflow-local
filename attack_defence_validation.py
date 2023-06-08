from art.attacks.evasion import CarliniL2Method, BasicIterativeMethod,FastGradientMethod,ProjectedGradientDescent,\
    MomentumIterativeMethod,NewtonFool,ProjectedGradientDescentPyTorch,UniversalPerturbation
from file_loader.bucket_loader import BucketLoader
from file_loader.file_handler import FileLoader
from user_files.helpers import get_files_package_root
class AttackDefenceValidator:
    def __init__(self, metadata):
        self.HyperParameters_ranges = {'CarliniL2Method': {'confidence': 0.2, "obj": CarliniL2Method},
                          'BasicIterativeMethod': {'eps': 0.3, 'eps_step': 0.15, 'max_iter': 10000,"obj": BasicIterativeMethod},
                          'FastGradientMethod': {'eps': 0.005,"obj": FastGradientMethod},
                          'ProjectedGradientDescent': {'eps': 0.03, 'max_iter': 10000,"obj":ProjectedGradientDescent},
                          'MomentumIterativeMethod': {'eps': 0.1,"obj":MomentumIterativeMethod},
                          'NewtonFool': {'eta': 0.035,"obj":NewtonFool},
                          'ProjectedGradientDescentPyTorch': {'eps': 0.2, 'random_eps': True,"obj": ProjectedGradientDescentPyTorch
                                                              },
                          'UniversalPerturbation': {'attacker': 'deepfool', 'delta': 0.5,"obj": UniversalPerturbation}}

        self.__file_loader = FileLoader(metadata)
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
        file_name = "attack_defence_metadata.json"
        attack_defence_metadata = self.__file_loader.get_file(file_name)
        attack_metadata = attack_defence_metadata['attack']
        estimator = self.__file_loader.get_estimator()
        for attack, bool_val in attack_metadata.items():
            if bool_val:
                attack_obj = self.HyperParameters_ranges[attack]["obj"]
                attack_estimator_req = attack_obj._estimator_requirements
                if not issubclass(estimator,attack_estimator_req):
                    attack_metadata[attack] = False

        attack_defence_metadata['attack'] = attack_metadata
        dest_path = get_files_package_root() + "/" + file_name
        self.__file_loader.save_file(obj=attack_defence_metadata,
                                     path=dest_path, as_json=True)








