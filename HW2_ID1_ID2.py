import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        self.w = None
        self.ids = (318880754, 206567067)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        :return: True if the perceptron found a linear separating hyperplane, False otherwise.
        """
        flag = True
        K = len(np.unique(y))
        dim = X.shape[1]
        w = [np.zeros(dim) for _ in range(K)]
        while flag:
            flag = False
            for i in np.arange(X.shape[0]):
                y_pred = np.argmax(np.inner(w, X[i]))
                yt = y[i]
                if y_pred != yt:
                    flag = True
                    for j in range(K):
                        if j != yt and j != y_pred:
                            w[yt] = w[yt] + X[i]
                            w[y_pred] = w[y_pred] - X[i]
        self.w = w

    def predict(self, X: np.ndarray) -> np.ndarray:

        y_pred = []
        for x in X:
            y_pred.append(np.argmax(np.inner(self.w, x)))
        return np.array(y_pred, dtype=np.uint8)


if __name__ == "__main__":
    print("*" * 20)
    print("Started HW2_318880754_206567067.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)
