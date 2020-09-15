# coding: utf-8

from utils import safe_opener
from sklearn.model_selection import train_test_split
from multilayerperceptron import *

csv_file = "resources\\data.csv"


def training():
    """
    Function used to train the Multilayer Perceptron model.
    
    :return: 
    """
    np.random.seed(42)
    data = safe_opener(csv_file)

    # Replacing y_test values by booleans 0 -> B | 1 -> M and the , by .
    data = data.replace(',', '.', regex=True)
    data = data.replace('B', '0', regex=True)
    data = data.replace('M', '1', regex=True)

    y = data.iloc[:,1]
    X = data.drop(data.columns[1], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Normalize
    X_test = X_test.to_numpy()
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Flatten
    X_train = X_train.reshape([X_train.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])

    network = []
    network.append(Dense(X_train.shape[1], 20))
    network.append(ReLU())
    network.append(Dense(20, 40))
    network.append(ReLU())
    network.append(Dense(40, 2))
    train(network, X_train, y_train)


if __name__ == '__main__':
    training()
