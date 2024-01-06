#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from imblearn.over_sampling import ADASYN
import pandas as pd

def adasyn(df : pd.DataFrame):
    '''
        Wrapper function for ADASYN
    '''
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print('Before ADASYN')
    print(y.value_counts())
    X_resampled, y_resampled = ADASYN().fit_resample(X, y)
    print('After ADASYN')
    print(y_resampled.value_counts())

    # Save the resampled dataset
    resampled_df = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    del(df)
    del(X)
    del(y)
    return resampled_df
