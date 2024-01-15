#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Sun Apr 16 18:59:57 2023.

@author: 
"""


def normalize(dlist):
    """Normalize a list of DataFrames by keeping only common columns.

    Parameters
    ----------
    dlist : list of pd.DataFrame
        List of DataFrames to be normalized.

    Returns
    -------
    dnew : list of pd.DataFrame
        List of normalized DataFrames with only common columns.
    """
    cols = []
    dnew = []

    for df in dlist:
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
        cols.append(df.columns)

    # Create sets of columns for each DataFrame
    cols_sets = [set(sublist) for sublist in cols]

    # Find the common columns across all DataFrames
    common = list(set(list(set.intersection(*cols_sets))))

    for df in dlist:
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
        df = df[df.columns.intersection(common)]  # Keep only common columns
        dnew.append(df)

    del dlist
    del cols_sets
    del common

    return dnew
