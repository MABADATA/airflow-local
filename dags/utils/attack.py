import time
from .attack_helpers import *
from .estimator_helpers import *
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

    (x_train, y_train),(x_test, y_test) = get_data()
    x_train = np.transpose(x_train, (0, 1)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 1)).astype(np.float32)
    estimator = get_estimator()
    estimator.fit(x_train,y_train)
    # x_test, y_test = next(data)
    T = time.time()
    logging.info("Optimizing...")
    optimized_attack = optimize_evasion_attack(attack_obj,estimator,x_test,y_test)
    logging.info(f"Optimizing done! optimization time is: {time.time() - T}")

    # attack_params = attack_parameter_range[attack_obj.__name__]
    # wrap_attack = set_attack_params(attack_obj,attack_params,estimator)
    # x_test, y_test = next(data)
    adversarial_examples = optimized_attack.generate(np.asarray(x_test))
    prediction_softmax_results = estimator.predict(adversarial_examples)
    prediction_results = np.argmax(prediction_softmax_results, axis=1)
    # y_train, y_test = next(data)
    # y_test = next(data)
    model_acc = accuracy_score(y_test, prediction_results)
    sent_model_after_attack(estimator)
    return model_acc, adversarial_examples
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
    model_acc, adversarial_examples = attack(LaserAttack)
    ti.xcom_push(key='attack_LaserAttack_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_LaserAttack_score', value=model_acc)

def attack_OverTheAirFlickeringPyTorch(ti):
    model_acc, adversarial_examples  = attack(OverTheAirFlickeringPyTorch)
    ti.xcom_push(key='attack_OverTheAirFlickeringPyTorch_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_OverTheAirFlickeringPyTorch_score', value=model_acc)

def attack_BasicIterativeMethod(ti):
    ti.xcom_push(key='Module_BasicIter', value=__name__)
    model_acc, adversarial_examples = attack(BasicIterativeMethod)
    ti.xcom_push(key='attack_BasicIterativeMethod_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_BasicIterativeMethod_score', value=model_acc)

def attack_FastGradientMethod(ti):
    ti.xcom_push(key='Module_FG', value=__name__)
    model_acc, adversarial_examples = attack(FastGradientMethod)
    ti.xcom_push(key='attack_FastGradientMethod_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_FastGradientMethod_score', value=model_acc)

def attack_ProjectedGradientDescent(ti):
    model_acc,adversarial_examples = attack(ProjectedGradientDescent)
    ti.xcom_push(key='attack_ProjectedGradientDescent_adv', value=adversarial_examples.tolist())
    ti.xcom_push(key='attack_ProjectedGradientDescent_score', value=model_acc)

# if __name__ == '__main__':
#     attack(ProjectedGradientDescent)