import pandas as pd
from NSGA_2 import nsga_2
import matplotlib.pyplot as plt
import gc

gc.enable()


def feature_modeling(feature_list):
    """Flatten a nested list of features.

    Parameters:
    - feature_list (list): Nested list of features.

    Returns:
    - flat_list (list): Flattened list of features.
    """
    flat_list = []
    for sublist in feature_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def global_feature_select(feature_list, num_ftr=0):
    """Perform global feature selection using FCMI and aFFMI scores.

    Parameters:
    - feature_list (list): List of features with corresponding FCMI and aFFMI scores.
    - num_ftr (int): Number of features to select (default is 0, meaning all features).

    Returns:
    - selected_features (list): List of selected features.
    - num_avbl_ftrs (int): Number of available features after selection.
    """
    flat_list = feature_modeling(feature_list)
    df = pd.DataFrame(flat_list, columns=["features", "FCMI", "aFFMI"])
    df = df.groupby("features").mean().reset_index()

    # Displaying global feature list using a scatter plot
    plt.scatter(df["FCMI"], df["aFFMI"])
    plt.xlabel("FCMI")
    plt.ylabel("aFFMI")
    plt.title("Global Feature list")
    for i in range(len(df)):
        plt.text(df["FCMI"][i], df["aFFMI"][i], df["features"][i])
    plt.show()

    ftrs_in_fronts = nsga_2(df)
    lst = []
    for front in ftrs_in_fronts:
        for i in front:
            lst.append(i)

    num_avbl_ftrs = len(lst)

    if num_ftr:
        return lst[:num_ftr], num_avbl_ftrs
    else:
        return lst, num_avbl_ftrs


def global_feature_select_single(feature_list, num_ftr=0):
    """Perform global feature selection prioritizing features with greater FCMI
    Score.

    Parameters:
    - feature_list (list): List of features with corresponding FCMI and aFFMI scores.
    - num_ftr (int): Number of features to select (default is 0, meaning all features).

    Returns:
    - selected_features (list): List of selected features.
    - num_avbl_ftrs (int): Number of available features after selection.
    """
    flat_list = feature_modeling(feature_list)
    df = pd.DataFrame(flat_list, columns=["features", "FCMI", "aFFMI"])
    df = df.groupby("features").mean().reset_index()

    df["FCMI"] = df["FCMI"] - (df["aFFMI"] / (len(df)))

    df.sort_values(by=["FCMI"], inplace=True, ascending=False)
    list1 = df["features"].values.tolist()

    num_avbl_ftrs = len(list1)

    if num_ftr:
        list1 = list1[:num_ftr]

    return list1, num_avbl_ftrs
