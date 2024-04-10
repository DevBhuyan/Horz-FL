import gc
import pandas as pd

gc.enable()


def fast_non_dominated_sort(df_copy):
    """Code based on NSGA-II by Deb et.Al.

    Refer: "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" by Kalyanmoy Deb
    """
    l = df_copy.shape[0]
    df_copy = df_copy.sort_values("FCMI").reset_index(drop=True)
    fronts = []

    for sweep in range(l):
        rem = len(df_copy)
        if rem:
            front = []
            prev = df_copy.iloc[rem - 1, :]
            front.append(prev)
            df_copy.drop(index=rem - 1, inplace=True)
            df_copy.reset_index(drop=True, inplace=True)
            for i in range(rem - 2, -1, -1):
                feature_tuple = df_copy.iloc[i, :]
                if feature_tuple["aFFMI"] < prev["aFFMI"]:
                    front.append(feature_tuple)
                    prev = feature_tuple
                    df_copy.drop(index=i, inplace=True)
                    df_copy.reset_index(drop=True, inplace=True)
            fronts.append(front)
        else:
            break
    del df_copy
    return fronts


def nsga_2(df: pd.DataFrame):
    """Version of NSGA 2 algorithm optimized to our use case.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    ftrs_in_fronts : list
        pareto fronts containing features.
    """

    # Initialization
    df_copy = df.copy(deep=True)

    non_dominated_sorted_solution = fast_non_dominated_sort(df_copy)

    ftrs_in_fronts = []
    for front in non_dominated_sorted_solution:
        ftrs_in_front = []
        for df in front:
            ftrs_in_front.append(df["features"])
        ftrs_in_fronts.append(ftrs_in_front)

    del df
    del non_dominated_sorted_solution

    return ftrs_in_fronts
