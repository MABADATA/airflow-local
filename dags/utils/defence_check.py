
def to_defence_HighConfidence(ti):
    to_attack = ti.xcom_pull(key='defence_HighConfidence',
                             task_ids=f'choose_defence')

    if to_attack:
        return 'defence_HighConfidence'
    else:
        return 'default'

def to_defence_ReverseSigmoid(ti):
    to_attack = ti.xcom_pull(key='defence_ReverseSigmoid',
                             task_ids=f'choose_defence')

    if to_attack:
        return 'defence_ReverseSigmoid'
    else:
        return 'default'
def to_defence_Rounded(ti):
    to_attack = ti.xcom_pull(key='defence_Rounded',
                             task_ids=f'choose_defence')

    if to_attack:
        return 'defence_Rounded'
    else:
        return 'default'

def to_defence_GaussianNoise(ti):
    to_attack = ti.xcom_pull(key='defence_GaussianNoise',
                             task_ids=f'choose_defence')

    if to_attack:
        return 'defence_GaussianNoise'
    else:
        return 'default'