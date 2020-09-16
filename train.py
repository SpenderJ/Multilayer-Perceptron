# coding: utf-8

from utils import safe_opener, data_manipulation
from multilayerperceptron import *
from tqdm import trange
from joblib import dump

import matplotlib.pyplot as plt
import os

csv_file = "resources\\data.csv"


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Train the model using minibatches

    :param inputs: entry dataset
    :param targets: target dataset
    :param batchsize: number of element
    :param shuffle: boolean for shuffling or no
    :return: 2 lists
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def training():
    """
    Function used to train the Multilayer Perceptron model.
    
    :return: Nothing, just creates a file with the model parameters
    """
    np.random.seed(42)
    data = safe_opener(csv_file)
    X_train, X_test, y_train, y_test = data_manipulation(data, False)

    # Creation of the network

    network = []
    network.append(Dense(X_train.shape[1], 20))
    network.append(ReLU())
    network.append(Dense(20, 40))
    network.append(ReLU())
    network.append(Dense(40, 2))

    # Train the model + Perfs and graphs

    train_log = []
    val_log = []

    for epoch in range(70):
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=32, shuffle=True):
            train(network, x_batch, y_batch)

        train_log.append(np.mean(predict(network, X_train) == y_train))
        val_log.append(np.mean(predict(network, X_test) == y_test))

        print("Epoch", epoch)
        print("Train accuracy:", train_log[-1])
        print("Val accuracy:", val_log[-1])
    plt.plot(train_log, label='train accuracy')
    plt.plot(val_log, label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # Remove the file if already existing

    try:
        os.remove('model.joblib')
    except OSError:
        pass
    dump(network, 'model.joblib')
    pass


if __name__ == '__main__':
    training()
