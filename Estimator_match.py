from Bucket_loader import Bucket_loader
import json
class Estimator_handler:
    """
    Attributes:
    ----------
    file_loader = Bucket_loader object.
    ml_type: str, classification/regression.
    implementation: str, the way the model was implemented(support pytorch, tensorflow and sklearn).
    algorithm: which ML algorithm the model perform(e.g logistic regression) only relevant to sklearn implementations.
    input_shape: tuple/list, shape of the data
    input_val_range: tuple/list, range of values the data gets( it can be tuple of tuples regarding columns).
    num_of_classes: int, the number of classes in a classification problem.
    estimator: estimator type(from art lib)  # get a value after self._wrap_model will run.
    map: dict, maps all objects base on the input above.
    Methods:
    _______
    estimator_match(): mapping an estimator to a ML model form art lib base on params.
    wrap_model(): wraps the ML model in the estimator.
    wrap(): serves as main that match, wrap and upload to GCP.
    """
    def __init__(self, input):
        self.__file_loader = Bucket_loader()
        if isinstance(self.__file_loader.metadata, str):
            self.metadata = json.loads(self.__file_loader.metadata)
        if isinstance(self.__file_loader.metadata, dict):
            self.metadata = self.__file_loader.metadata
        else:
            raise TypeError('meta data need to be type dict or str')

        self.__ml_type = input["ML_type"]
        self.__implementation = input["implementation"]
        self.__algorithm = input['algorithm']
        self.__input_shape = self.metadata['ML_model']['input']['input_shape']
        self.__input_val_range = self.metadata['ML_model']['input']["input_val_range"]
        self.__num_of_classes = self.metadata['ML_model']['input']["num_of_classes"]
        self.estimator = None  # get a value after self._wrap_model will run
        self.map = {"implementation": {
            "pytorch": {
                "ML_type": {
                    "classification": "PyTorchClassifier",
                    "regression": "PyTorchRegressor"
                }
            },
            "tensorflow": {
                "ML_type": {
                    "classification": "TensorFlowV2Classifier",
                    "regression": "KerasRegressor"

                }
            },
            "sklearn": {
                "ML_type": {
                    "classification": {
                        "DecisionTree": "ScikitlearnDecisionTreeClassifier",
                        "ExtraTree": "ScikitlearnExtraTreesClassifier",
                        "AdaBoost": "ScikitlearnAdaBoostClassifier",
                        "Bagging": "ScikitlearnBaggingClassifier",
                        "GradientBoosting": "ScikitlearnGradientBoostingClassifier",
                        "RandomForest": "ScikitlearnRandomForestClassifier",
                        "LogisticRegression": "ScikitlearnLogisticRegression",
                        "GaussianNB": "ScikitlearnGaussianNB",
                        "SVC": "ScikitlearnSVC",
                        "LinearSVC": "ScikitlearnSVC",
                    },
                    "regression": {
                        "DecisionTree":  "ScikitlearnDecisionTreeRegressor"}}
            }
        }

        }

    def _estimator_match(self):
        if self.__implementation == 'sklearn':
            estimator = self.map['implementation']['sklearn']['ML_type'][self.__ml_type][self.__algorithm]
        else:
            estimator = self.map['implementation'][self.__implementation]['ML_type'][self.__ml_type]
        return estimator

    def _estimator_params(self):
        param_dict = {'loss': False, 'optimizer': False, 'clip_values': None, 'nb_classes': None, 'input_shape': None}
        if self.__implementation == 'sklearn':
            param_dict['clip_values'] = self.__input_val_range
        elif self.__implementation == 'tensorflow':
            if self.__ml_type == "regression":
                pass
            elif self.__ml_type == "classification":
                param_dict['nb_classes'] = self.__num_of_classes
                param_dict['input_shape'] = self.__input_shape
            else:
                raise Exception("Can't wrap model!\nML type must be classification or regression")
        elif self.__implementation == 'pytorch':
            if self.metadata['ML_model']['optimizer']['meta']['optimizer_type']:
                param_dict['optimizer'] = True
            param_dict['loss'] = True
            if self.__ml_type == "regression":
                param_dict['input_shape'] = self.__input_shape

            elif self.__ml_type == "classification":
                param_dict['input_shape'] = self.__input_shape
                param_dict['nb_classes'] = self.__num_of_classes

            else:
                raise Exception("Can't wrap model!\nML type must be classification or regression")

        else:
            raise Exception("Can't wrap model!\nImplementation type must be sklearn,pytorch or tensorflow")
        #extract the params that needed - meaning not None
        param_dict = {k: v for k, v in param_dict.items() if v}
        return param_dict


    def wrap(self):
        estimator_obj = self._estimator_match()
        params = self._estimator_params()
        estimator_dict ={"object": estimator_obj, "prams": params}
        with open("Estimator_params.json", 'w') as f:
            json.dump(estimator_dict, f)

        self.__file_loader.upload(obj=estimator_dict, obj_type="Estimator_params", to_pickle=False)
