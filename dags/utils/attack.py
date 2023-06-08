import time
from .attack_helpers import *
from file_loader.file_handler import *
from user_files.model.model_def import *
from user_files.dataloader.dataloader_def import *
from art.attacks.evasion import SquareAttack,\
    TargetedUniversalPerturbation,\
    UniversalPerturbation,\
    VirtualAdversarialMethod,Wasserstein,ZooAttack,\
    FrameSaliencyAttack,GeoDA,ElasticNet,CarliniL2Method,BoundaryAttack,\
    AutoProjectedGradientDescent,DeepFool,AutoAttack,LowProFool,NewtonFool,\
    MalwareGDTensorFlow,PixelAttack,SaliencyMapMethod,ShadowAttack,\
    SpatialTransformation,ShapeShifter,SignOPTAttack,AdversarialPatch,AdversarialPatchPyTorch,\
    FeatureAdversariesPyTorch,GRAPHITEBlackbox,GRAPHITEWhiteboxPyTorch,LaserAttack,\
    OverTheAirFlickeringPyTorch, BasicIterativeMethod, FastGradientMethod, ProjectedGradientDescent

def attack(attack_obj):
    return 0, np.array([1,2,3,4])
    # (x_train, y_train),(x_test, y_test) = get_data()
    # x_train = np.transpose(x_train, (0, 1)).astype(np.float32)
    # x_test = np.transpose(x_test, (0, 1)).astype(np.float32)
    # estimator = get_estimator()
    # estimator.fit(x_train,y_train)
    # # x_test, y_test = next(data)
    # T = time.time()
    # logging.info("Optimizing...")
    # optimized_attack = optimize_evasion_attack(attack_obj,estimator,x_test,y_test)
    # logging.info(f"Optimizing done! optimization time is: {time.time() - T}")
    #
    # # attack_params = attack_parameter_range[attack_obj.__name__]
    # # wrap_attack = set_attack_params(attack_obj,attack_params,estimator)
    # # x_test, y_test = next(data)
    # adversarial_examples = optimized_attack.generate(np.asarray(x_test))
    # prediction_softmax_results = estimator.predict(adversarial_examples)
    # prediction_results = np.argmax(prediction_softmax_results, axis=1)
    # # y_train, y_test = next(data)
    # # y_test = next(data)
    # model_acc = accuracy_score(y_test, prediction_results)
    # sent_model_after_attack(estimator)
    # return model_acc, adversarial_examples


#attack here