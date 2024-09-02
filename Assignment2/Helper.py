import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


def special_split(X, y, ratio=0.7):
    """
    This function splits the dataset into training and test sets so that the missing data is only in the training set.
    Make it easier to preprocess in this assignment
    :param X: feature matrix, must be Panda dataframe
    :param y: label vector, must be Panda dataframe
    :param ratio: testing ratio, default is 0.7
    :return: split training and test sets
    """
    n_i, n_f = X.shape
    n_train = int(0.7 * n_i)
    n_test = n_i - n_train

    null_rows = pd.isna(X).any(axis=1)
    null_rows = np.where(null_rows)[0]
    non_null_rows = np.setdiff1d(np.arange(0, n_i), null_rows)
    X_non_null, y_non_null = X.iloc[non_null_rows], y.iloc[non_null_rows]
    X_train, X_test, y_train, y_test = train_test_split(X_non_null, y_non_null, test_size=n_test, random_state=231)
    X_train, y_train = pd.concat([X_train, X.iloc[null_rows]]), pd.concat([y_train, y[null_rows]])
    train_per = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train.iloc[train_per], y_train.iloc[train_per]
    return X_train, X_test, y_train, y_test


def evaluation(clf, X_train, y_train, X_test, y_test):
    """
    Calculate the performance of a the classifier trained on the training set (X_train, y_train)
    and test the test set (X_test, y_test).
    :param clf: a classifier
    :param X_train: training feature matrix, must be numpy array
    :param y_train: training label vector, must be numpy array
    :param X_test: testing feature matrix, must be numpy array
    :param y_test: testing label vector, must be numpy array
    :return: classification accuracy
    """
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    acc = balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
    return acc
