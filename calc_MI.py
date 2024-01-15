import numpy as np
import pandas as pd
import gc

gc.enable()


def mutual_information(X, Y, bins):
    """Calculate the Mutual Information between two variables.

    Parameters
    ----------
    X, Y : array-like
        Input variables for which Mutual Information is calculated.
    bins : int
        Number of bins for histogram calculation.

    Returns
    -------
    MI : float
        Mutual Information between X and Y.
    """
    # Create a 2D histogram for X and Y
    c_XY = np.histogram2d(X, Y, bins)[0]
    # Histogram of common occurrences of the 1st feature (X)
    c_X = np.histogram(X, bins)[0]
    # Histogram of common occurrences of the 2nd feature (Y)
    c_Y = np.histogram(Y, bins)[0]

    # Calculate Shannon entropy for X, Y, and joint distribution XY
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    # Calculate Mutual Information
    MI = H_X + H_Y - H_XY
    return MI


def shan_entropy(c):
    """Calculate the Shannon entropy of a distribution.

    Parameters
    ----------
    c : array-like
        Input distribution.

    Returns
    -------
    H : float
        Shannon entropy.
    """
    # Normalize the distribution to obtain probabilities
    c_normalized = c / float(np.sum(c))
    # Retain non-zero values only
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    # Calculate Shannon entropy
    H = -sum(c_normalized * np.log2(c_normalized))
    return H


def calc_MI(df: pd.DataFrame):
    """Calculate Mutual Information for all pairs of features in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    df1 : pd.DataFrame
        DataFrame containing Mutual Information values for all feature pairs.
    """
    cols = df.columns
    bins = 10
    A = df.to_numpy()
    n = A.shape[1]
    matMI = np.zeros((n, n))

    # Iterate over all pairs of features and calculate Mutual Information
    for ix in np.arange(n):
        for jx in np.arange(0, n):
            if ix == jx:
                matMI[ix, jx] = 0  # Mutual Information with itself is zero
            else:
                matMI[ix, jx] = mutual_information(A[:, ix], A[:, jx], bins)

    print("PRINTING MAT_MI")
    print(matMI)
    print(matMI.shape)

    # Create a DataFrame with Mutual Information values
    df1 = pd.DataFrame(matMI, columns=cols)
    return df1
