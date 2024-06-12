#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fed_mrmr import federated_mrmr
from local_feature_select import local_fs, full_spec_fs
from global_feature_select import global_feature_select, global_feature_select_single
from horz_data_divn import horz_data_divn, CLIENT_DIST_FOR_NONIID, NUM_CLASSES
from ff import ff
from Fed_MLP import Fed_MLP
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os
from warnings import warn
from tqdm import tqdm
from joblib import Parallel, delayed


N_JOBS = 4
warn(f"Using {N_JOBS if N_JOBS > 0 else 'all'} out of 12 cores for RFE. Update N_JOBS in ff_helpers.py as required")


def run_noniid(n_clust_fcmi: int,
               n_clust_ffmi: int,
               dataset: str,
               num_ftr: int,
               obj: str,
               classifier: str,
               iid_ratio: float = 1.0):
    """
    Run the non-IID scenario for feature selection and classification.

    Parameters:
    - n_clust_fcmi (int): Number of clusters for local feature selection using FCMi.
    - n_clust_ffmi (int): Number of clusters for local feature selection using FFmi.
    - dataset (str): Name of the dataset.
    - num_ftr (int): Number of features to select.
    - obj (str): Feature selection objective ('single', 'multi', 'anova', 'rfe', 'mrmr').
    - classifier (str): The classifier to use ('ff' for Federated Forest, 'mlp' for Federated MLP).
    - iid_ratio (float, optional): Ratio for IID data. Defaults to 1.0.

    Returns:
    Tuple[str, float, float]: Tuple containing the classifier name, accuracy, and F1 score.
    """

    local_feature = []
    os.makedirs('./dataframes_to_send', exist_ok=True)

    n_client = CLIENT_DIST_FOR_NONIID[dataset]

    df_list = horz_data_divn(
        dataset, n_client, non_iid=True, iid_ratio=iid_ratio)

    # The following code stores the local_features into a variable lftr so that they need not be explicitly computed for different num_ftr during a single run
    if os.path.exists(f"./lftr_cache/lftr_{n_clust_fcmi}_{n_clust_ffmi}_{dataset}.pkl"):
        with open(f"./lftr_cache/lftr_{n_clust_fcmi}_{n_clust_ffmi}_{dataset}.pkl", "rb") as f:
            warn("Data loaded from cache. Delete ./lftr_cache to run local_fs again")
            lftr = pickle.load(f)
    else:
        if obj in ["single", "multi"]:
            print("Generating Local Feature set")
            for cli in tqdm(range(0, n_client), total=n_client):
                data_dfx = df_list[cli]
                if dataset == "vowel" or dataset == "vehicle":
                    local = full_spec_fs(
                        data_dfx, n_clust_fcmi, n_clust_ffmi)
                else:
                    local = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi)
                # local_feature.append(local)
                warn("`local_feature.append(local)` was deprecated in version 2.0. Now lftr doesn't contain client wise feature_scores")
                local_feature.extend(local)
            lftr = local_feature
            if not os.path.exists('./lftr_cache'):
                os.makedirs('./lftr_cache', exist_ok=True)
            with open(f"./lftr_cache/lftr_{n_clust_fcmi}_{n_clust_ffmi}_{dataset}.pkl", "wb") as f:
                pickle.dump(lftr, f)

    if obj == "single":
        # Single-Objective ftr sel
        return single_obj(lftr, num_ftr, n_client, df_list, classifier, NUM_CLASSES[dataset], iid_ratio, dataset)

    elif obj == "multi":
        # Multi-Objective ftr sel
        return multi_obj(lftr, num_ftr, n_client, df_list, classifier, NUM_CLASSES[dataset], iid_ratio, dataset)

    elif obj == "anova":
        # ANOVA
        return anova_obj(num_ftr, n_client, classifier, df_list, NUM_CLASSES[dataset], iid_ratio, dataset)

    elif obj == "rfe":
        # RFE
        return rfe_obj(num_ftr, n_client, classifier, df_list, NUM_CLASSES[dataset], iid_ratio, dataset)

    elif obj == 'mrmr':
        # Fed-mRMR
        return mrmr_obj(num_ftr, n_client, df_list, classifier, NUM_CLASSES[dataset], iid_ratio, dataset)


def single_obj(lftr,
               num_ftr,
               n_client,
               df_list,
               classifier,
               num_classes, iid_ratio, dataset):

    storage_file = f"./dataframes_to_send/df_list_for_single_{n_client}_{num_ftr}_{iid_ratio}_{dataset}.pkl"

    if not os.path.exists(storage_file):
        feature_list, num_avbl_ftrs = global_feature_select_single(
            lftr, num_ftr, iid_ratio, dataset, n_client)

        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class=df1)
            dataframes_to_send.append(data_dfx)

        with open(storage_file, "wb") as f:
            pickle.dump(dataframes_to_send, f)

    else:
        with open(storage_file, "rb") as f:
            dataframes_to_send = pickle.load(f)

    return classify(classifier, dataframes_to_send, num_classes)


def multi_obj(lftr,
              num_ftr,
              n_client,
              df_list,
              classifier,
              num_classes, iid_ratio, dataset):

    storage_file = f"./dataframes_to_send/df_list_for_multi_{n_client}_{num_ftr}_{iid_ratio}_{dataset}.pkl"

    if not os.path.exists(storage_file):
        feature_list, num_avbl_ftrs = global_feature_select(
            lftr, num_ftr, iid_ratio, dataset, n_client)

        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class=df1)
            dataframes_to_send.append(data_dfx)

        with open(storage_file, "wb") as f:
            pickle.dump(dataframes_to_send, f)

    else:
        with open(storage_file, "rb") as f:
            dataframes_to_send = pickle.load(f)

    return classify(classifier, dataframes_to_send, num_classes)


def anova_obj(num_ftr,
              n_client,
              classifier,
              df_list,
              num_classes, iid_ratio, dataset):

    storage_file = f"./dataframes_to_send/df_list_for_anova_{n_client}_{num_ftr}_{iid_ratio}_{dataset}.pkl"

    if not os.path.exists(storage_file):
        f_selector = SelectKBest(score_func=f_classif, k=num_ftr)

        dataframes_to_send = []
        for df in tqdm(df_list, desc='Performing ANOVA FS....', total=len(df_list)):
            df = df.reset_index(drop=True)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            f_selector.fit_transform(X, y)
            feature_indices = f_selector.get_support(indices=True)
            selected_feature_names = X.columns[feature_indices]
            df = pd.DataFrame(X)
            df = df[df.columns.intersection(selected_feature_names)]
            df = df.assign(Class=y)
            dataframes_to_send.append(df)

        with open(storage_file, "wb") as f:
            pickle.dump(dataframes_to_send, f)

    else:
        with open(storage_file, "rb") as f:
            dataframes_to_send = pickle.load(f)

    return classify(classifier, dataframes_to_send, num_classes)


def _rfe_process_df(df, estimator, num_ftr):
    df = df.reset_index(drop=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    rfe = RFE(estimator, n_features_to_select=num_ftr)
    X = rfe.fit_transform(X, y)
    df = pd.DataFrame(X)
    df = df.assign(Class=y)
    return df


def rfe_obj(num_ftr,
            n_client,
            classifier,
            df_list,
            num_classes,
            iid_ratio,
            dataset):

    storage_file = f"./dataframes_to_send/df_list_for_rfe_{n_client}_{num_ftr}_{iid_ratio}_{dataset}.pkl"

    if not os.path.exists(storage_file):
        estimator = RandomForestClassifier()

        # dataframes_to_send = Parallel(
        # n_jobs=N_JOBS)(delayed(_rfe_process_df)(df, estimator, num_ftr) for df in df_list)
        dataframes_to_send = []
        for df in tqdm(df_list, total=len(df_list)):
            dataframes_to_send.append(_rfe_process_df(df, estimator, num_ftr))

        with open(storage_file, "wb") as f:
            pickle.dump(dataframes_to_send, f)

    else:
        with open(storage_file, "rb") as f:
            dataframes_to_send = pickle.load(f)

    return classify(classifier, dataframes_to_send, num_classes)


def mrmr_obj(num_ftr,
             n_client,
             df_list,
             classifier,
             num_classes,
             iid_ratio,
             dataset):

    storage_file = f"./dataframes_to_send/df_list_for_mrmr_{n_client}_{num_ftr}_{iid_ratio}_{dataset}.pkl"

    if not os.path.exists(storage_file):
        feature_list = federated_mrmr(df_list, num_ftr, iid_ratio, dataset)

        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class=df1)
            dataframes_to_send.append(data_dfx)

        with open(storage_file, "wb") as f:
            pickle.dump(dataframes_to_send, f)

    else:
        with open(storage_file, "rb") as f:
            dataframes_to_send = pickle.load(f)

    return 0, 0, 0

    # return classify(classifier, dataframes_to_send, num_classes)


def classify(classifier: str,
             dataframes_to_send: list,
             num_classes: int,
             non_iid: bool = True):
    """
    Classify the data using the specified classifier.

    Parameters:
    - classifier (str): The classifier to use ('ff' for Federated Forest, 'mlp' for Federated MLP).
    - dataframes_to_send (list): List of dataframes.
    - num_classes (int): Number of classes in the dataset.
    - non_iid (bool, optional): Whether the data is non-IID. Defaults to True.

    Returns:
    Tuple[str, float, float]: Tuple containing the classifier name, accuracy, and F1 score.
    """

    print("\nTraining on:", classifier)
    if classifier == "ff":
        ff_acc, ff_f1 = ff(dataframes_to_send, num_classes, non_iid=True)
        # print(f"ff_acc: {ff_acc}, ff_f1: {ff_f1}")
        return "ff", ff_acc, ff_f1
    elif classifier == "mlp":
        MLP_acc, MLP_f1 = Fed_MLP(dataframes_to_send, num_classes=num_classes)
        # print(f"MLP_acc: {MLP_acc}, MLP_f1: {MLP_f1}")
        return "mlp", MLP_acc, MLP_f1
