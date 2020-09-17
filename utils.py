# coding: utf-8

import os
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def data_manipulation(data, prediction):
    """
    All data manipulation used on the original CSV to make my life easier and the code cleaner

    :param data: Data Frame containing the original CSV
    :param prediction: boolean to check if we are predicting
    :return: 4 numpy arrays containing my data or 1 numpy array
    """
    data = data.replace(',', '.', regex=True)
    data = data.replace('B', '0', regex=True)
    data = data.replace('M', '1', regex=True)

    if not prediction:
        y = data.iloc[:, 1]
        X = data.drop(data.columns[1], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Normalize
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train.values)
        X_test = min_max_scaler.fit_transform(X_test.values)
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Flatten
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

        return X_train, X_test, y_train, y_test
    elif prediction:
        y = data.iloc[:, 1]
        X = data.drop(data.columns[1], axis=1)

        y = y.to_numpy().astype(int)

        # Normalize
        min_max_scaler = MinMaxScaler()
        X = min_max_scaler.fit_transform(X.values)
        X = X.astype(float)
        X = X.reshape([X.shape[0], -1])
        return X, y


def safe_opener(file):
    """
    Function used to safely open the csv file

    :param file: name of the file
    :return data: containing the training set
    """
    cwd = os.getcwd()
    print(cwd)
    try:
        f = open(os.path.join(cwd, file), 'rb')
        encoding = chardet.detect(f.read())
        data = pd.read_csv(os.path.join(cwd, file), encoding=encoding['encoding'], delimiter=';')
    except Exception as e:
        print("Cant open the csv passed as argument")
        raise e
    return data
