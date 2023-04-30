
def to_attack_SquareAttack(ti):
    to_attack = ti.xcom_pull(key='attack_SquareAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SquareAttack'
    else:
        return 'default_attack'
def to_attack_TargetedUniversalPerturbation(ti):
    to_attack = ti.xcom_pull(key='attack_TargetedUniversalPerturbation',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_TargetedUniversalPerturbation'
    else:
        return 'default_attack'
def to_attack_UniversalPerturbation(ti):
    to_attack = ti.xcom_pull(key='attack_UniversalPerturbation',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_UniversalPerturbation'
    else:
        return 'default_attack'
def to_attack_VirtualAdversarialMethod(ti):
    to_attack = ti.xcom_pull(key='attack_VirtualAdversarialMethod',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_VirtualAdversarialMethod'
    else:
        return 'default_attack'
def to_attack_Wasserstein(ti):
    to_attack = ti.xcom_pull(key='attack_Wasserstein',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_Wasserstein'
    else:
        return 'default_attack'
def to_attack_ZooAttack(ti):
    to_attack = ti.xcom_pull(key='attack_ZooAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ZooAttack'
    else:
        return 'default_attack'
def to_attack_FrameSaliencyAttack(ti):
    to_attack = ti.xcom_pull(key='attack_FrameSaliencyAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_FrameSaliencyAttack'
    else:
        return 'default_attack'
def to_attack_GeoDA(ti):
    to_attack = ti.xcom_pull(key='attack_GeoDA',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_GeoDA'
    else:
        return 'default_attack'
def to_attack_ElasticNet(ti):
    to_attack = ti.xcom_pull(key='attack_ElasticNet',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ElasticNet'
    else:
        return 'default_attack'
def to_attack_CarliniL2Method(ti):
    to_attack = ti.xcom_pull(key='attack_CarliniL2Method',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_CarliniL2Method'
    else:
        return 'default_attack'
def to_attack_BoundaryAttack(ti):
    to_attack = ti.xcom_pull(key='attack_BoundaryAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_BoundaryAttack'
    else:
        return 'default_attack'
def to_attack_AutoProjectedGradientDescent(ti):
    to_attack = ti.xcom_pull(key='attack_AutoProjectedGradientDescent',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_AutoProjectedGradientDescent'
    else:
        return 'default_attack'
def to_attack_DeepFool(ti):
    to_attack = ti.xcom_pull(key='attack_DeepFool',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_DeepFool'
    else:
        return 'default_attack'
def to_attack_AutoAttack(ti):
    to_attack = ti.xcom_pull(key='attack_AutoAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_AutoAttack'
    else:
        return 'default_attack'
def to_attack_LowProFool(ti):
    to_attack = ti.xcom_pull(key='attack_LowProFool',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_LowProFool'
    else:
        return 'default_attack'
def to_attack_NewtonFool(ti):
    to_attack = ti.xcom_pull(key='attack_NewtonFool',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_NewtonFool'
    else:
        return 'default_attack'
def to_attack_MalwareGDTensorFlow(ti):
    to_attack = ti.xcom_pull(key='attack_MalwareGDTensorFlow',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_MalwareGDTensorFlow'
    else:
        return 'default_attack'
def to_attack_PixelAttack(ti):
    to_attack = ti.xcom_pull(key='attack_PixelAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_PixelAttack'
    else:
        return 'default_attack'
def to_attack_SaliencyMapMethod(ti):
    to_attack = ti.xcom_pull(key='attack_SaliencyMapMethod',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SaliencyMapMethod'
    else:
        return 'default_attack'
def to_attack_ShadowAttack(ti):
    to_attack = ti.xcom_pull(key='attack_ShadowAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ShadowAttack'
    else:
        return 'default_attack'
def to_attack_SpatialTransformation(ti):
    to_attack = ti.xcom_pull(key='attack_SpatialTransformation',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SpatialTransformation'
    else:
        return 'default_attack'
def to_attack_ShapeShifter(ti):
    to_attack = ti.xcom_pull(key='attack_ShapeShifter',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ShapeShifter'
    else:
        return 'default_attack'
def to_attack_SignOPTAttack(ti):
    to_attack = ti.xcom_pull(key='attack_SignOPTAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_SignOPTAttack'
    else:
        return 'default_attack'
def to_attack_AdversarialPatch(ti):
    to_attack = ti.xcom_pull(key='attack_AdversarialPatch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_AdversarialPatch'
    else:
        return 'default_attack'
def to_attack_AdversarialPatchPyTorch(ti):
    to_attack = ti.xcom_pull(key='attack_AdversarialPatchPyTorch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_AdversarialPatchPyTorch'
    else:
        return 'default_attack'
def to_attack_FeatureAdversariesPyTorch(ti):
    to_attack = ti.xcom_pull(key='attack_FeatureAdversariesPyTorch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_FeatureAdversariesPyTorch'
    else:
        return 'default_attack'
def to_attack_GRAPHITEBlackbox(ti):
    to_attack = ti.xcom_pull(key='attack_GRAPHITEBlackbox',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_GRAPHITEBlackbox'
    else:
        return 'default_attack'
def to_attack_GRAPHITEWhiteboxPyTorch(ti):
    to_attack = ti.xcom_pull(key='attack_GRAPHITEWhiteboxPyTorch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_GRAPHITEWhiteboxPyTorch'
    else:
        return 'default_attack'
def to_attack_LaserAttack(ti):
    to_attack = ti.xcom_pull(key='attack_LaserAttack',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_LaserAttack'
    else:
        return 'default_attack'
def to_attack_OverTheAirFlickeringPyTorch(ti):
    to_attack = ti.xcom_pull(key='attack_OverTheAirFlickeringPyTorch',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_OverTheAirFlickeringPyTorch'
    else:
        return 'default_attack'
def to_attack_BasicIterativeMethod(ti):
    to_attack = ti.xcom_pull(key='attack_BasicIterativeMethod',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_BasicIterativeMethod'
    else:
        return 'default_attack'

def to_attack_FastGradientMethod(ti):
    to_attack = ti.xcom_pull(key='attack_FastGradientMethod',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_FastGradientMethod'
    else:
        return 'default_attack'
def to_attack_ProjectedGradientDescent(ti):
    to_attack = ti.xcom_pull(key='attack_ProjectedGradientDescent',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_ProjectedGradientDescent'
    else:
        return 'default_attack'



