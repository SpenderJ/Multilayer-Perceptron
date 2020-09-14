# coding: utf-8

from utils import safe_opener
from sklearn.model_selection import train_test_split

csv_file = "resources\\data.csv"


def train():
    """
    Function used to train the Multilayer Perceptron model.
    
    :return: 
    """
    data = safe_opener(csv_file)
    X = data.iloc[1]
    y = data.drop(data.columns[1], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


if __name__ == '__main__':
    train()
