# coding: utf-8

from utils import safe_opener
from sklearn.model_selection import train_test_split
from multilayerperceptron import *
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
import matplotlib.pyplot as plt

csv_file = "resources\\data.csv"


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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

    network = []
    network.append(Dense(X_train.shape[1], 20))
    network.append(ReLU())
    network.append(Dense(20, 40))
    network.append(ReLU())
    network.append(Dense(40, 2))

    # Pred part

    train_log = []
    val_log = []

    for epoch in range(25):
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


if __name__ == '__main__':
    training()
