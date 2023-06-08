import os

def get_model_package_root() -> str:
    """Returns the root directory of your project."""
    return os.path.dirname(os.path.abspath(__file__))