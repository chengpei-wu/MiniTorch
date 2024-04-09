import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from mt.auto_grad.tensor import Tensor
from mt.data.dataset import Dataset

__all__ = [
    'load_dataset'
]


def load_minst():
    df = pd.read_csv(f"{os.path.dirname(__file__)}/dataset/minst.csv")
    labels = df.loc[:, "label"].values
    imgs = df.iloc[:, 1:].values
    scaler = MinMaxScaler()
    imgs = scaler.fit_transform(imgs.T).T
    labels = list(map(int, labels.reshape((len(labels),)).tolist()))
    # one-hot encode
    labels = np.eye(10)[labels]
    X, y = np.array(imgs), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return Dataset(Tensor(X_train), Tensor(y_train)), Dataset(Tensor(X_test), Tensor(y_test))


def load_dataset(dataset_name='minst'):
    if dataset_name == 'minst':
        return load_minst()
    elif dataset_name == 'titanic':
        raise NotImplementedError(f'dataset {dataset_name}')
    else:
        raise NotImplementedError(f'dataset {dataset_name}')


if __name__ == '__main__':
    load_minst()
