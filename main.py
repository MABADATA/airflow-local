from Input_validation import InputValidator
from Env_prep import Envsetter
from Estimator_match import EstimatorHandler
from attack_defence_validation import  AttackDefenceValidator


# metadata = {
#     "ML_model": {
#         "meta": {
#             "file_id": "ML_file",
#             "model_type": "sklearn",
#             "ML_type": "classification",
#             "algorithm": "randomForest"
#         },
#         "input": {
#             "input_shape": (100, 20),
#             "input_val_range": (10, 20),
#             "num_of_classes": None
#         },
#         "loss": {
#             "meta": {
#                 "file_id": None,
#                 "loss_type": None
#             }
#         },
#         "optimizer": {
#             "meta": {
#                 "file_id": None,
#                 "optimizer_type": None
#             }
#         }
#     },
#     "dataloader": {
#         "meta": {
#             "file_id": "Dataloader_file",
#             "data_loader_type": "pandad.DataFrame"
#         }
#     },
#     "requirements.txt": {
#         "meta": {
#             "file_id": "requierments.txt"
#         }
#
#     },
#     "authentication": {
#         "bucket_name": "tomer-test",
#         "access_key_id": "AKIA5YFFMYPZGIEL6PFP",
#         "secret_access_key": "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",
#         "region": "us-east-1"
#     }
# }

def validation_process(params_from_api):
    metadata = (dict(params_from_api))
    env_setter = Envsetter(metadata)
    env_setter.install_requirements()
    attack_defence_validator = AttackDefenceValidator(metadata)
    attack_defence_validator.validate()
    input_validator = InputValidator(metadata)
    if input_validator.validate():
        input = input_validator.get_input()
        wrapper = EstimatorHandler(input, metadata)
        wrapper.wrap()
        return True
    return False

if __name__ == '__main__':
    param = {}
    validation_process(param)
