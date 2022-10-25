import numpy as np
import pandas as pd


def mutual_information(X, Y, bins):
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H


def calc_MI(df):
    cols = df.columns
    bins = 10
    A = df.to_numpy()
    n = A.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(0, n):
            if ix == jx:
                matMI[ix, jx] = 0
            else:
                # print(ix)
                # print(jx)
                matMI[ix, jx] = mutual_information(A[:, ix], A[:, jx], bins)
    # print(matMI)
    df1 = pd.DataFrame(matMI, columns=cols)
    return df1
