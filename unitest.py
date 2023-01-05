import unittest
import torch.nn as nn
import tensorflow as tf
from Test_gen import Test
from sklearn.ensemble import (AdaBoostClassifier,BaggingClassifier,
                              GradientBoostingClassifier,RandomForestClassifier)
from sklearn.base import BaseEstimator
from torch.optim import Optimizer,SGD
from torch.nn.modules.loss import _Loss
from keras.losses import Loss
class MyTestCase(unittest.TestCase):
    def test_generate_model(self):
        T1 = Test(model_type='tensorflow', dataloader_type='numpy_array')
        T2 = Test(model_type='pytorch', dataloader_type='numpy_array')
        T3 = Test(model_type='sklearn', dataloader_type='numpy_array', algorithm='AdaBoost')
        assert isinstance(T1.gen_model(), tf.keras.models.Model) == True
        assert isinstance(T2.gen_model(), nn.Module) == True
        assert isinstance(T3.gen_model(), AdaBoostClassifier) == True
        assert isinstance(T3.gen_model(), BaseEstimator) == True
    def test_generate_optimizer(self):
        T1 = Test(model_type='pytorch', dataloader_type='numpy_array', optimizer=True)
        T2 = Test(model_type='pytorch', dataloader_type='numpy_array', optimizer=False)
        T3 = Test(model_type='pytorch', dataloader_type='numpy_array', optimizer=True, valid=False)
        assert isinstance(T1.gen_optimizer(), Optimizer) == True
        assert isinstance(T2.gen_optimizer(), Optimizer) == False
        assert isinstance(T3.gen_optimizer(), Optimizer) == False

    def test_generate_loss_func(self):
        T1 = Test(model_type='pytorch', dataloader_type='numpy_array', loss=True)
        T2 = Test(model_type='tensorflow', dataloader_type='numpy_array', loss=True)
        T3 = Test(model_type='tensorflow', dataloader_type='numpy_array', loss=False)
        T4 = Test(model_type='tensorflow', dataloader_type='numpy_array', loss=True, valid=False)
        assert isinstance(T1.gen_loss(), _Loss) == True
        assert isinstance(T2.gen_loss(), Loss) == True
        assert isinstance(T3.gen_loss(), Loss) == False
        assert isinstance(T4.gen_loss(), Loss) == False
    def test_flow(self):
        T = Test(model_type='pytorch', dataloader_type='array', loss=True, nb_classes=True)
        T.run_test()
    def test_open(self):
        open('requirements.txt','r')


if __name__ == '__main__':
    unittest.main()
