import logging
from .helpers import *
from torch.optim import SGD
import torch.nn as nn
def assign_vars(cls, args_dict,ML_model):
    if args_dict.get("optimizer"):
        optimizer = SGD(ML_model.parameters(), lr=0.01)
        args_dict["optimizer"] = optimizer
    if args_dict.get("loss"):
        try:
            loss = load_from_bucket('loss.pickle')
        # only from test
            args_dict["loss"] = loss
        except:
            args_dict["loss"] = nn.CrossEntropyLoss()
    obj = cls(**args_dict,model=ML_model)
    return obj
def get_obj_from_str(obj_as_str):
    estimator_map = {
        "PyTorchClassifier": PyTorchClassifier,
        "PyTorchRegressor": PyTorchRegressor,
        "TensorFlowV2Classifier": TensorFlowV2Classifier,
        "KerasRegressor": KerasRegressor,
        "ScikitlearnDecisionTreeClassifier": ScikitlearnDecisionTreeClassifier,
        "ScikitlearnExtraTreesClassifier": ScikitlearnExtraTreesClassifier,
        "ScikitlearnAdaBoostClassifier": ScikitlearnAdaBoostClassifier,
        "ScikitlearnBaggingClassifier": ScikitlearnBaggingClassifier,
        "ScikitlearnGradientBoostingClassifier": ScikitlearnGradientBoostingClassifier,
        "ScikitlearnRandomForestClassifier": ScikitlearnRandomForestClassifier,
        "ScikitlearnLogisticRegression": ScikitlearnLogisticRegression,
        "ScikitlearnGaussianNB": ScikitlearnGaussianNB,
        "ScikitlearnSVC": ScikitlearnSVC,
        "ScikitlearnDecisionTreeRegressor": ScikitlearnDecisionTreeRegressor
    }
    return estimator_map[obj_as_str]

def get_estimator():
    logging.info("Getting estimator...")
    # args = load_from_bucket("Estimator_params.json", as_json=True)
    args = {"object": "PyTorchClassifier", "params": {"loss": True, "optimizer": True, "nb_classes": 2, "input_shape":(1,29)}}
    model = load_from_bucket("ML_model.pickle")
    # model = NeuralNetworkClassificationModel(29,2)
    estimator_str = args['object']
    estimator_obj = get_obj_from_str(estimator_str)
    params = args['params']
    estimator = assign_vars(cls=estimator_obj,args_dict=params,ML_model=model)
    return estimator

# if __name__ == '__main__':
#     print(get_estimator())