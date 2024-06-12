#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from imblearn.over_sampling import SMOTE
import pandas as pd


def smote(df: pd.DataFrame,
          verbose: bool = False):
    """
    Implementation of SMOTE (Synthetic Minority Over-sampling Technique).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing features and labels.
    verbose : bool, optional
        If True, prints the class distribution before and after applying SMOTE. Default is False.

    Returns
    -------
    resampled_df : pd.DataFrame
        DataFrame with resampled data using SMOTE.
    """

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    smote = SMOTE()
    if verbose:
        print("Before SMOTE")
        print(y.value_counts())
    X_resampled, y_resampled = smote.fit_resample(X, y)
    if verbose:
        print("After SMOTE")
        print(y_resampled.value_counts())

    resampled_df = pd.concat(
        [pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1
    )

    del df
    del X
    del y
    return resampled_df
