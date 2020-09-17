# coding: utf-8

import pandas as pd
import numpy as np
import sys
from joblib import load
import os
from utils import safe_opener, data_manipulation
from multilayerperceptron import predict as prediction
from multilayerperceptron import predict_probas
from sklearn.metrics import confusion_matrix, roc_auc_score

pred_file = "resources\\template.csv"
model_file = "model.joblib"


def cross_entropy_loss(probas, y):
    log_likelihood = -np.log(probas[range(y.shape[0]), y])
    loss = np.sum(log_likelihood) / y.shape[0]
    return loss


def predict():
    """
    Function used to predict using the given dataset in argument

    :return:
    """
    pred = safe_opener(pred_file)

    cwd = os.getcwd()
    print(cwd)
    try:
        f = open(os.path.join(cwd, model_file), 'rb')
        network = load(cwd + '\\' + model_file)
    except Exception as e:
        print("Cant open the model file")
        raise e
    X, y = data_manipulation(pred, True)
    preds = prediction(network, X)
    probas = predict_probas(network, X)

    # Results
    tn, fp, fn, tp = confusion_matrix(np.argmax(probas, axis=1), y).ravel()
    print('\nConfusion matrix: \n', confusion_matrix(np.argmax(probas, axis=1), y))
    print('Accuracy: {0:.4f}%'.format(((tn + tp) / y.shape[0])*100))
    print('ROC AUC score: {0:.2f}'.format(roc_auc_score(y, np.argmax(probas, axis=1))))
    print('Cross entropy loss: {0:.4f}\n'.format(cross_entropy_loss(probas, y)*100))
    pass


if __name__ == '__main__':
    predict()
