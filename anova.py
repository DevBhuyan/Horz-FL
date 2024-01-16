#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
from sklearn.feature_selection import f_classif, SelectKBest
from tqdm import tqdm


def anova(df_list: list, num_ftr: int):
    """Wrapper function for ANOVA FS method.

    Parameters
    ----------
    df_list : list
    num_ftr : int

    Returns
    -------
    new_list : TYPE
        new list of reduced dataframes.
    """
    start = datetime.now()

    f_selector = SelectKBest(score_func=f_classif, k=num_ftr)

    new_list = []
    for df in tqdm(df_list, total=len(df_list)):
        df = df.reset_index(drop=True)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        f_selector.fit_transform(X, y)
        feature_indices = f_selector.get_support(indices=True)
        selected_feature_names = X.columns[feature_indices]
        df = pd.DataFrame(X)
        df = df[df.columns.intersection(selected_feature_names)]
        df = df.assign(Class=y)
        new_list.append(df)

    print(
        "\033[1;33m"
        + f"Total learning time : {(datetime.now()-start)/len(df_list)}"
        + "\033[0m"
    )
    return new_list
