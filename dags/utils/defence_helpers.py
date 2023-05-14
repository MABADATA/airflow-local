from sklearn.metrics import accuracy_score
from skopt import gp_minimize
import numpy as np
from skopt.space import Integer
from helpers import get_data
from estimator_helpers import get_estimator
from art.defences.postprocessor import GaussianNoise,  Rounded, ReverseSigmoid, HighConfidence,Postprocessor
from helpers import load_from_bucket, upload_to_bucket

def pars_defence_json(ti):
    # json_data = get_json_from_bucket()
    # defence_json = json.loads(load_from_bucket('attack_defence_metadata.json'))['defence']
    ti.xcom_push(key='Module_pars', value=__name__)
    defence_json = {'GaussianNoise': True, 'ReverseSigmoid': True, 'Rounded': True,
                                         'HighConfidence': True}
    for defence, bool_val in defence_json.items():
        ti.xcom_push(key=defence, value=bool_val)


def set_defence_params(defence, params,estimator ):
    defence_with_params = defence(**params,estimator=estimator)
    return defence_with_params

def choose_best_defences(ti):

    defence_score_pair = {'GaussianNoise': 0,'ReverseSigmoid': 0, 'Rounded': 0, 'HighConfidence': 0}
    defence_name_obj_pair = {'GaussianNoise': GaussianNoise,
                             'ReverseSigmoid': ReverseSigmoid, 'Rounded': Rounded,
                             'HighConfidence': HighConfidence}
    defence_HP_pair = {'GaussianNoise': None,'ReverseSigmoid': None, 'Rounded': None, 'HighConfidence': None}
    for index, defence_score in enumerate(defence_score_pair):  # attack score is the key
        defence_dict = ti.xcom_pull(key=defence_score + "_score",
                             task_ids=f'{defence_score}')
        if defence_dict:
            defence_score_pair[defence_score] = defence_dict['accuracy']
    best_defence = max(defence_score_pair, key=defence_score_pair.get)
    best_score = defence_score_pair[best_defence]
    best_defence_HP = defence_HP_pair[best_defence]
    best_defence_obj = defence_name_obj_pair[best_defence](**best_defence_HP)
    metadata = load_from_bucket(file_name='attack_defence_metadata.json', as_json=True)
    metadata['cycles'] += 1
    metadata['defence_best_scores'].append((best_defence,best_score))
    estimator_param_dict = load_from_bucket(file_name="Estimator_params.json",as_json=True)
    estimator_param_dict['postprocessing_defences'].append(b)
    ti.xcom_push(key=f'best defence {best_defence}', value=best_score)
    upload_to_bucket(obj=metadata, file_name='attack_defence_metadata.json', as_json=True)
    return

def get_defenses_hyperparameters():
    return  {GaussianNoise: ['scale'], ReverseSigmoid: ['beta'], Rounded: ['decimals'],
     HighConfidence: ['cutoff']}
def get_defenses_hyperparameters_ranges():
    return {GaussianNoise: {'scale': (0.00001, 1)},
            ReverseSigmoid: {'beta': (0.00001, 1)},
            Rounded: {'decimals': Integer(1, 5)},
            HighConfidence: {'cutoff': (0.000001, 1)}}


def optimize_post_processor_optimization(classifier,defense, adv_examples,true_labels):
    """
    This function only responsible for optimize the hyper parameters a post processor type defense.
    :param defense: the defense we wish to optimize its hyper paraemeters.
    :param adv_examples: the adversarial examples we want to defend from.
    :return: the optimized defense instance.
    """

    def _optimize(defense_to_optimize, hyperparams):
        if (isinstance(hyperparams[0], np.int64)):  # Only for the Rounded hyperparamer - it doesnt work with
            # np.int64.
            hyperparams[0] = int(hyperparams[0])
        defenses_hyperparameters = get_defenses_hyperparameters()
        HP = {k: v for k, v in zip(defenses_hyperparameters[defense], hyperparams)}
        defense_to_optimize = defense_to_optimize(**HP)
        classifier.postprocessing_defences = [
            defense_to_optimize]  # each time we will add the next defense on to
        prediction_softmax_results_def = classifier.predict(adv_examples)
        prediction_results_def = np.argmax(prediction_softmax_results_def, axis=1)
        # get the model's accuracy after the denfese applied and return
        acc = accuracy_score(true_labels, prediction_results_def)
        return -acc

    defenses_hyperparameters_ranges = get_defenses_hyperparameters_ranges()
    defenses_hyperparameters = get_defenses_hyperparameters()
    search_space = [v for k, v in defenses_hyperparameters_ranges[defense].items()]
    func = lambda params: _optimize(defense, params)
    result = gp_minimize(func, search_space, n_calls=10)
    final_HP = {k: v for k, v in zip(defenses_hyperparameters[defense], result.x)}
    optimize_defence = defense(**final_HP)
    return optimize_defence, final_HP

def clean_model_defenses(classifier):
    """
    cleaning the classifier defenses lists in order to isolate only the tested defense.
    """
    classifier.postprocessing_defences = []
    classifier.preprocessing_defenses = []

def try_defense(classifier,defense, adv_examples,true_labels):
    """
    This function is checking the target model's accuracy when defense is applied
    :param defense: the defense we are applying on the model
    :param adv_examples: The adversarial examples we want to defende from
    :return: the target model accuracy and the optimized defense instance.
    """
    clean_model_defenses(classifier)
    optimized_defense, optimized_defense_HP = optimize_post_processor_optimization(classifier,defense, adv_examples,true_labels)
    if isinstance(optimized_defense, Postprocessor):
        classifier.postprocessing_defences.append(
            optimized_defense)
    else:
        classifier.preprocessing_defenses.append(
            optimized_defense)  # each time we will add the next defense on top of the last one.
    prediction_softmax_results_def = classifier.predict(adv_examples)
    prediction_results_def = np.argmax(prediction_softmax_results_def, axis=1)
    # get the model's accuracy after the denfese applied and return it.
    acc = accuracy_score(true_labels, prediction_results_def)
    clean_model_defenses(classifier)
    return acc, optimized_defense_HP

def select_best_PostProcessor_defense(classifier, possible_defenses,adv_examples, k,true_labels):
    # tuple that holds the defense that got the best accuracy against those adv exmaples.
    # classifier_defenses = copy.deepcopy(self.classifier.postprocessing_defences)
    # print([type(i).__name__ for i in classifier_defenses])
    top_k_acc_defenses = []  # the top k chosen defenses.
    # defense in terms of model accruacy.
    for post_defense in possible_defenses['Post Processors']:
        acc, opt_defense = try_defense(classifier=classifier,
                                       defense=post_defense,
                                            adv_examples=adv_examples,
                                       true_labels=true_labels)  # Trying each defense in order to get the top K
        top_k_acc_defenses.append((acc, opt_defense))
    # self.classifier.postprocessing_defences = classifier_defenses
    top_k_acc_defenses = sorted(top_k_acc_defenses, key=lambda x: x[0], reverse=True)  # Reverse beacuse we
    # want the accuracy to be the highest.
    top_k = [i[1] for i in top_k_acc_defenses][:k]  # get only the defenses sorted by the model accuracy.
    classifier.postprocessing_defences = top_k  # apply the defenses on the model.

    # Get the model accuracy with the chosen k defenses.
    final_pred = classifier.predict(adv_examples)
    prediction_results_final = np.argmax(final_pred, axis=1)
    final_acc = accuracy_score(true_labels, prediction_results_final)
    # defenses_names = [type(i).__name__ for i in self.classifier.postprocessing_defences]
    # if type(max_acc_defense['Defense']).__name__ in defenses_names:
    #     return max_acc_defense['model accuracy']
    # self.classifier.postprocessing_defences.append(max_acc_defense['Defense'])
    return final_acc

