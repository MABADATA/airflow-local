
from Preprocess.Bucket_loader import Bucket_loader

metadata = {
    "ML_model": {
        "meta": {
            "file_id": "ML_file",
            "model_type": "sklearn",
            "ML_type": "classification",
            "algorithm": "randomForest"
        },
        "input": {
            "input_shape": (100, 20),
            "input_val_range": (10, 20),
            "num_of_classes": None
        },
        "loss": {
            "meta": {
                "file_id": None,
                "loss_type": None
            }
        },
        "optimizer": {
            "meta": {
                "file_id": None,
                "optimizer_type": None
            }
        }
    },
    "dataloader": {
        "meta": {
            "file_id": "Dataloader_file",
            "data_loader_type": "pandad.DataFrame"
        }
    },
    "requirements": {
        "meta": {
            "file_id": "requirements.txt"
        }

    },
    "authentication": {
        "bucket_name": "tomer-test",
        "access_key_id": "AKIA5YFFMYPZGIEL6PFP",
        "secret_access_key": "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",
        "region": "us-east-1"
    }
}

if __name__ == '__main__':


    b = Bucket_loader(metadata)
    r = b.get_requirements()
