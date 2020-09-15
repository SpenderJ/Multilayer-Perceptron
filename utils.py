# coding: utf-8

import os
import chardet
import pandas as pd


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
