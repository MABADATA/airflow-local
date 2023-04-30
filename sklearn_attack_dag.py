from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from datetime import datetime
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from helpers import load_from_bucket, upload_to_bucket, get_data
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from art.attacks.evasion import ZooAttack
from art.estimators.classification import XGBoostClassifier
from art.utils import load_mnist

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 16),
    'retries': 1
}

def run_function(func_name, run_func):
    if run_func:
        # Run the function
        return f"Running function: {func_name}"
    else:
        return f"Not running function: {func_name}"


def pars_json(ti):
    # json_data = get_json_from_bucket()
    # attack_json = json.loads(load_from_bucket('attack.json'))
    attack_json = {'attack_zoo': True,
                   }
    for attack, bool_val in attack_json.items():
        ti.xcom_push(key=attack, value=bool_val)



def to_attack_zoo(ti):
    to_attack = ti.xcom_pull(key='attack_zoo',
                             task_ids=f'choose_attack')

    if to_attack:
        return 'attack_zoo'
    else:
        return 'default'


def attack_zoo(ti):
    model_acc = attack(ZooAttack)
    ti.xcom_push(key='attack_zoo_score', value=model_acc)


def get_fix_data():
    data = get_data()
    x_train, x_test, y_train,y_test = train_test_split(data)
    x_train = x_train[y_train < 2][:, [0, 1]]
    y_train = y_train[y_train < 2]
    x_train[:, 0][y_train == 0] *= 2
    x_train[:, 1][y_train == 2] *= 2
    x_train[:, 0][y_train == 0] -= 3
    x_train[:, 1][y_train == 2] -= 2

    x_train[:, 0] = (x_train[:, 0] - 4) / (9 - 4)
    x_train[:, 1] = (x_train[:, 1] - 1) / (6 - 1)

    return x_train, y_train


def attack(attack_obj):

    # Step 1: Load the MNIST dataset
    (x_train, y_train), (x_test, y_test),min_pixel_value, max_pixel_value = load_mnist()
    # Step 1a: Flatten dataset
    x_test = x_test[0:5]
    y_test = y_test[0:5]

    nb_samples_train = x_train.shape[0]
    nb_samples_test = x_test.shape[0]
    x_train = x_train.reshape((nb_samples_train, 28 * 28))
    x_test = x_test.reshape((nb_samples_test, 28 * 28))

    # Step 2: Create the model

    params = {"objective": "multi:softprob", "metric": "accuracy", "num_class": 10}
    dtrain = xgb.DMatrix(x_train, label=np.argmax(y_train, axis=1))
    dtest = xgb.DMatrix(x_test, label=np.argmax(y_test, axis=1))
    evals = [(dtest, "test"), (dtrain, "train")]
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=2, evals=evals)

    # Step 3: Create the ART classifier

    classifier = XGBoostClassifier(
        model=model, clip_values=(min_pixel_value, max_pixel_value), nb_features=28 * 28, nb_classes=10
    )
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples
    attack = attack_obj(
        classifier=classifier,
        confidence=0.0,
        targeted=False,
        learning_rate=1e-1,
        max_iter=200,
        binary_search_steps=10,
        initial_const=1e-3,
        abort_early=True,
        use_resize=False,
        use_importance=False,
        nb_parallel=5,
        batch_size=1,
        variable_h=0.01,
    )
    x_test_adv = attack.generate(x=x_test, y=y_test)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    return accuracy
# def attack(attack):
#     return random.randint(1, 10)
#     data = get_data()
#     estimator = load_from_bucket('estimator.pickle')
#     x_train = next(data)
#     adversarial_examples = attack.generate(x_train)
#     prediction_softmax_results = estimator.predict(adversarial_examples)
#     prediction_results = np.argmax(prediction_softmax_results, axis=1)
#     y_test = next(data)
#     model_acc = accuracy_score(y_test, prediction_results)
#     upload_to_bucket(estimator, 'estimator.pickle')
#     return model_acc
def default(ti):
    ti.xcom_push(key='Default', value=0)

def choose_best_attack(ti):

    attacks_scores = {'attack_BasicIterativeMethod_score':0,'attack_FastGradientMethod_score':0,'attack_ProjectedGradientDescent_score':0}
    for index,attacks_score in enumerate(attacks_scores):
        score = ti.xcom_pull(key=attacks_score,
                             task_ids=f'attack{index + 1}')
        if score:
            attacks_scores[attacks_score] = ti.xcom_pull(key=attacks_score,
                                 task_ids=f'attack{index + 1}')

    best_score = max(attacks_scores.values())
    metadata = load_from_bucket(file_name='attack_defence_metadata.json',as_json=True)
    for key, val in attacks_scores.items():
        if val == best_score:
            metadata['attack_best_scores'].append((key,val))
            ti.xcom_push(key=f'best attack {key} in round {metadata["cycles"]} : ', value=val)
            upload_to_bucket(obj=metadata,file_name='attack_defence_metadata.json',as_json=True)
            return
def set_or_create(ti):
    try:
        metadata = load_from_bucket(file_name='attack_defence_metadata.json',as_json=True)
        ti.xcom_push(key='metadata', value='exist')
    except:
        ti.xcom_push(key='metadata', value='not exist...creating....')
        metadata = {"cycles": 0, "attack_best_scores": [], "defence_best_scores": []}
        upload_to_bucket(obj=metadata, file_name='attack_defence_metadata.json',as_json=True)
        ti.xcom_push(key='metadata', value='uploaded')






with DAG('multi_sklearn_attack_dag', schedule_interval='@daily', default_args=default_args, catchup=False) as dag:

    choose_attack = PythonOperator(
            task_id=f'choose_attack',
            python_callable=pars_json
        )

    metadata = PythonOperator(
        task_id="metadata",
        python_callable=set_or_create
    )
    branch1 = BranchPythonOperator(
        task_id='to_attack_zoo',
        python_callable=to_attack_zoo
    )



    run_attack_zoo = PythonOperator(
            task_id=f"attack_zoo",
            python_callable=attack_zoo
        )


    run_default = PythonOperator(
            task_id=f"default",
            python_callable=default
        )
    choose_best = PythonOperator(
            task_id=f"choose_best",
            python_callable=choose_best_attack,
            trigger_rule='none_failed'
        )
    trigger_defence = TriggerDagRunOperator(
        task_id='trigger_defence',
        trigger_dag_id='multi_defence_dag'
    )
    #
    choose_attack >> metadata >> branch1 >> [run_attack_zoo, run_default] >> choose_best
    choose_best >> trigger_defence

