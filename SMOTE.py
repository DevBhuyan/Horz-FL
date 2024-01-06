#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from imblearn.over_sampling import SMOTE
import pandas as pd

def smote(df : pd.DataFrame):
    '''
    Implementation of SMOTE

    Parameters
    ----------
    df : pd.DataFrame
        raw dataframe.

    Returns
    -------
    resampled_df : pd.DataFrame
        SMOTE processed dataframe.

    '''
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    smote = SMOTE()
    print('Before SMOTE')
    print(y.value_counts())
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print('After SMOTE')
    print(y_resampled.value_counts())

    resampled_df = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    
    del(df)
    del(X)
    del(y)
    return resampled_df
