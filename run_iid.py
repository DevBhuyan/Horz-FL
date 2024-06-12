#!/usr/bin/env python
# -*- coding: utf-8 -*-


from fed_mrmr import federated_mrmr
from tqdm import tqdm
import pickle
import os
from warnings import warn
from local_feature_select import local_fs, full_spec_fs
from global_feature_select import global_feature_select, global_feature_select_single
from horz_data_divn import horz_data_divn, NUM_CLASSES
from ff import ff
from Fed_MLP import Fed_MLP
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


N_JOBS = 4
warn(f"Using {N_JOBS if N_JOBS > 0 else 'all'} out of 12 cores for RFE. Update N_JOBS in ff_helpers.py as required")


def run_iid(n_client: int,
            n_clust_fcmi: int,
            n_clust_ffmi: int,
            dataset: str,
            num_ftr: int,
            obj: str,
            classifier: str,
            max_depth: int = 200):

    local_feature = []
    os.makedirs('./dataframes_to_send', exist_ok=True)

    df_list = horz_data_divn(dataset,
                             n_client,
                             non_iid=False)

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
        return single_obj(lftr, num_ftr, n_client, df_list, classifier, NUM_CLASSES[dataset],  dataset, max_depth)

    elif obj == "multi":
        # Multi-Objective ftr sel
        return multi_obj(lftr, num_ftr, n_client, df_list, classifier, NUM_CLASSES[dataset], dataset, max_depth)

    elif obj == "anova":
        # ANOVA
        return anova_obj(num_ftr, n_client, classifier, df_list, NUM_CLASSES[dataset], dataset, max_depth)

    elif obj == "rfe":
        # RFE
        return rfe_obj(num_ftr, n_client, classifier, df_list, NUM_CLASSES[dataset], dataset, max_depth)

    elif obj == 'mrmr':
        # Fed-mRMR
        return mrmr_obj(num_ftr, n_client, df_list, classifier, NUM_CLASSES[dataset], dataset, max_depth)

    elif obj == 'nofs':
        # No-FS
        return nofs_obj(num_ftr, n_client, df_list, classifier, NUM_CLASSES[dataset], dataset, max_depth)


def single_obj(lftr,
               num_ftr,
               n_client,
               df_list,
               classifier,
               num_classes,
               dataset,
               max_depth):

    iid_ratio = 1.0

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

    return classify(classifier, dataframes_to_send, num_classes, non_iid=False, max_depth=max_depth)


def multi_obj(lftr,
              num_ftr,
              n_client,
              df_list,
              classifier,
              num_classes,
              dataset,
              max_depth):

    iid_ratio = 1.0

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

    return classify(classifier, dataframes_to_send, num_classes, non_iid=False, max_depth=max_depth)


def anova_obj(num_ftr,
              n_client,
              classifier,
              df_list,
              num_classes,
              dataset, max_depth):

    iid_ratio = 1.0

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

    return classify(classifier, dataframes_to_send, num_classes, non_iid=False, max_depth=max_depth)


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
            dataset,
            max_depth):

    iid_ratio = 1.0

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

    return classify(classifier, dataframes_to_send, num_classes, non_iid=False, max_depth=max_depth)


def mrmr_obj(num_ftr,
             n_client,
             df_list,
             classifier,
             num_classes,
             dataset,
             max_depth):

    iid_ratio = 1.0

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

    return classify(classifier, dataframes_to_send, num_classes, non_iid=False, max_depth=max_depth)


def nofs_obj(num_ftr,
             n_client,
             df_list,
             classifier,
             num_classes,
             dataset,
             max_depth):

    return classify(classifier, df_list, num_classes, non_iid=False, max_depth=max_depth)


def classify(classifier: str,
             dataframes_to_send: list,
             num_classes: int,
             non_iid: bool = False,
             max_depth: int = 200):

    print("\nTraining on:", classifier)
    if classifier == "ff":
        if max_depth == 200:
            ff_acc, ff_f1 = ff(dataframes_to_send,
                               num_classes,
                               non_iid=False)
            print(f"ff_acc: {ff_acc}, ff_f1: {ff_f1}")
            return "ff", ff_acc, ff_f1
        else:
            ff_acc, ff_f1, returned_max_depth, total_leaves = ff(dataframes_to_send,
                                                                 num_classes,
                                                                 non_iid=False,
                                                                 max_depth=max_depth)

            print(f"ff_acc: {ff_acc}, ff_f1: {ff_f1}")
            return "ff", ff_acc, ff_f1, returned_max_depth, total_leaves

    elif classifier == "mlp":
        MLP_acc, MLP_f1 = Fed_MLP(dataframes_to_send, num_classes=num_classes)
        print(f"MLP_acc: {MLP_acc}, MLP_f1: {MLP_f1}")
        return "mlp", MLP_acc, MLP_f1
