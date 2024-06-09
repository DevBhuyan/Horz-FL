import pandas as pd
from NSGA_2 import nsga_2
import matplotlib.pyplot as plt
from warnings import warn
import os
import pickle
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


def global_feature_select(feature_list: list,
                          num_ftr: int,
                          iid_ratio: float,
                          dataset: int,
                          n_client: int,
                          verbose: bool = False):
    # flat_list = feature_modeling(feature_list)
    # df = pd.DataFrame(flat_list, columns=["features", "FCMI", "aFFMI"])

    # TODO: Write code to fetch lst from storage, reuse code from horz_data_divn
    stored_file = f"./feature_lists/multi_obj_{dataset}_{n_client}_{str(iid_ratio)}.pkl"
    if os.path.exists(stored_file):
        with open(stored_file, "rb") as f:
            lst = pickle.load(f)
            warn("Data loaded from cache. Delete ./feature_lists to run afresh")
            if num_ftr:
                return lst[:num_ftr], len(lst)
            else:
                return lst, len(lst)

    warn("`flat_list` was deprecated in version 2.0 to improve efficiency")
    df = pd.DataFrame(feature_list, columns=["features", "FCMI", "aFFMI"])
    df = df.groupby("features").mean().reset_index()

    if verbose:
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

    # TODO: Write code to save lst to storage, create folder if DNE
    if not os.path.exists('./feature_lists'):
        os.makedirs('./feature_lists')
    with open(stored_file, 'wb') as f:
        pickle.dump(lst, f)

    if num_ftr:
        return lst[:num_ftr], len(lst)
    else:
        return lst, len(lst)


def global_feature_select_single(feature_list,
                                 num_ftr,
                                 iid_ratio,
                                 dataset,
                                 n_client):
    # flat_list = feature_modeling(feature_list)
    # df = pd.DataFrame(flat_list, columns=["features", "FCMI", "aFFMI"])

    stored_file = f"./feature_lists/single_obj_{dataset}_{n_client}_{str(iid_ratio)}.pkl"
    if os.path.exists(stored_file):
        with open(stored_file, "rb") as f:
            lst = pickle.load(f)
            warn("Data loaded from cache. Delete ./feature_lists to run afresh")
            if num_ftr:
                return lst[:num_ftr], len(lst)
            else:
                return lst, len(lst)

    warn("`flat_list` was deprecated in version 2.0 to improve efficiency")
    df = pd.DataFrame(feature_list, columns=["features", "FCMI", "aFFMI"])
    df = df.groupby("features").mean().reset_index()

    df["FCMI"] = df["FCMI"] - (df["aFFMI"] / (len(df)))

    df.sort_values(by=["FCMI"], inplace=True, ascending=False)
    lst = df["features"].values.tolist()

    if not os.path.exists('./feature_lists'):
        os.makedirs('./feature_lists')
    with open(stored_file, 'wb') as f:
        pickle.dump(lst, f)

    if num_ftr:
        return lst[:num_ftr], len(lst)
    else:
        return lst, len(lst)
