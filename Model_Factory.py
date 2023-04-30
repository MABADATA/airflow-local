import sklearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from art.estimators.classification import PyTorchClassifier
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch import nn
import pickle


class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_dime, output_dime):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.input_layer = nn.Linear(input_dime, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        # self.hidden_layer2  = nn.Linear(24,20)
        # self.hidden_layer3  = nn.Linear(20,24)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(64, output_dime)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        # out =  self.relu(self.hidden_layer2(out))
        # out =  self.relu(self.hidden_layer3(out))

        out = self.output_layer(out)
        out = self.sigmoid(out)

        return out

class ModelFactory:
    def __init__(self, data_name):
        self.name = data_name

    def __make_trained_CC_model(self,model):
        data = pd.read_csv('creditcard.csv')
        for col in data:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        ss = StandardScaler()
        data["NormalizedAmount"] = ss.fit_transform(data["Amount"].values.reshape(-1, 1))
        data = data.drop(["Amount"], axis=1)
        data = data.drop(["Time"], axis=1)
        data = data.dropna()
        Y = data["Class"].values
        X = data.drop(["Class"], axis=1).values
        # The data is very imbalance so this is how i choosed t balance it.
        # ros = RandomOverSampler(random_state=0)
        ros = RandomUnderSampler(random_state=0)
        X, Y = ros.fit_resample(X, Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        model_CC = model
        learning_rate = 0.01
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_CC.parameters(), lr=learning_rate)
        return X, Y, optimizer
        X_train = np.transpose(X_train, (0, 1)).astype(np.float32)
        X_test = np.transpose(X_test, (0, 1)).astype(np.float32)
        classifier_CC = PyTorchClassifier(
            model=model_CC,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 29),
            nb_classes=2,

        )
        classifier_CC.fit(X_train, y_train, batch_size=32, nb_epochs=20)
        return classifier_CC, X_test, y_test

    def __make_trained_heart_model(self):
        heart_df = pd.read_csv('heart.csv')
        heart_df.drop_duplicates()
        y_heart = heart_df.output.to_numpy()
        X_heart = heart_df.drop('output', axis=1).to_numpy()
        scaler = StandardScaler()
        X_heart = scaler.fit_transform(X_heart)
        X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.2,
                                                                                    stratify=y_heart)
        # create data loaders
        batch_size = 5
        input_size = 13  # number of features
        output_size = 2
        model_heart = NeuralNetworkClassificationModel(input_size, output_size)
        loss_fn = nn.CrossEntropyLoss()  # Binary Cross Entropy
        optim = torch.optim.Adam(model_heart.parameters(), lr=1e-3)
        X_train_heart = np.transpose(X_train_heart, (0, 1)).astype(np.float32)
        X_test_heart = np.transpose(X_test_heart, (0, 1)).astype(np.float32)

        classifier_heart = PyTorchClassifier(
            model=model_heart,
            loss=loss_fn,
            optimizer=optim,
            input_shape=(1, 13),
            nb_classes=2,

        )
        classifier_heart.fit(X_train_heart, y_train_heart)
        return classifier_heart, X_test_heart,y_test_heart

    def make_trained_model(self,model):
        if self.name == 'CC':
            clf, data, true_labels = self.__make_trained_CC_model(model)
        else:
            clf, data, true_labels = self.__make_trained_heart_model()
        return clf, data, true_labels

if __name__ == '__main__':
    s = ModelFactory("CC")
    input_dim = 29
    output_dim = 2
    model_CC = NeuralNetworkClassificationModel(input_dim, output_dim)
    data,true_labels, optimizer = s.make_trained_model(model_CC)
    # print(clf)
    x = pd.DataFrame(data)
    y = pd.DataFrame(true_labels)
    # torch.save(model,)

    from helpers import upload_to_bucket
    upload_to_bucket(obj=model_CC, file_name="dags/ML_model.pickle")
    # upload_to_bucket(obj=optimizer,file_name="optimizer.pickle")
    # upload_to_bucket(x,"X.csv",as_csv=True)
    # upload_to_bucket(y, "y.csv", as_csv=True)
    loss = nn.CrossEntropyLoss()
    # upload_to_bucket(obj=loss, file_name="loss.pickle")
