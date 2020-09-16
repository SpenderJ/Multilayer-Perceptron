# coding: utf-8

import pandas as pd
import numpy as np
import sys
from joblib import load
import os
from utils import safe_opener, data_manipulation
from multilayerperceptron import predict as prediction

pred_file = "resources\\template.csv"
model_file = "model.joblib"


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
    X = data_manipulation(pred, True)
    y = prediction(network, X)
    print(y)
    pass


if __name__ == '__main__':
    predict()
