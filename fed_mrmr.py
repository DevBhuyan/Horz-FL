from datetime import datetime
from horz_data_divn import MAX_FTRS
import os
import pickle
from tqdm import tqdm
from horz_data_divn import horz_data_divn
import numpy as np
from sklearn.metrics import mutual_info_score as mutual_info


def local_mrmr(df,
               num_ftr,
               dataset):
    candidates = list(df.columns)[:-1]

    if dataset not in ["california", "boston"]:
        df['Class'] = df['Class'].astype(int)

    relevances = np.array([mutual_info(np.nan_to_num(df[i].values),
                                       np.nan_to_num(df['Class'].values))
                           for i in candidates])

    return relevances


def precompute_mutual_info(df_list):
    mutual_info_scores = {}
    features = list(df_list[0].columns)[:-1]

    for feature in tqdm(features, desc="Precomputing mutual-information", total=len(features)):
        for other_feature in features:
            if feature != other_feature and (feature, other_feature) not in mutual_info_scores:
                mutual_info_score_value = mutual_info(
                    np.nan_to_num(df_list[0][feature].values),
                    np.nan_to_num(df_list[0][other_feature].values))

                mutual_info_scores[(feature, other_feature)
                                   ] = mutual_info_score_value
                mutual_info_scores[(other_feature, feature)
                                   ] = mutual_info_score_value

    return mutual_info_scores


def federated_mrmr(df_list,
                   num_ftr,
                   iid_ratio,
                   dataset):
    n_client = len(df_list)
    aggregated_relevances = None

    storage_file = f'./feature_lists/mrmr_obj_{dataset}_{n_client}_{iid_ratio}.pkl'

    if os.path.exists(storage_file):
        with open(storage_file, 'rb') as f:
            all_features_ranked = pickle.load(f)
            return all_features_ranked[:num_ftr]

    for df in tqdm(df_list, desc="Computing relevances for mRMR", total=n_client):
        relevances = local_mrmr(df, num_ftr, dataset)

        if aggregated_relevances is None:
            aggregated_relevances = relevances
        else:
            aggregated_relevances += relevances

    aggregated_relevances /= n_client

    mutual_info_scores = precompute_mutual_info(df_list)

    all_features_ranked = []
    candidates = list(df_list[0].columns)[:-1]

    best_feature = candidates[np.argmax(aggregated_relevances)]
    all_features_ranked.append(best_feature)
    candidates.remove(best_feature)

    with tqdm(total=MAX_FTRS[dataset], desc="Selecting features by Fed-mRMR") as pbar:
        while len(all_features_ranked) < MAX_FTRS[dataset]:

            max_mrmr = -np.inf
            best_feature = -1

            for feature in candidates:
                relevance = aggregated_relevances[candidates.index(feature)]
                redundancy = np.mean([mutual_info_scores[(feature, f)]
                                     for f in all_features_ranked])
                mrmr = relevance - redundancy

                if mrmr > max_mrmr:
                    max_mrmr = mrmr
                    best_feature = feature

            all_features_ranked.append(best_feature)
            candidates.remove(best_feature)

            pbar.update(1)

    with open(storage_file, 'wb') as f:
        pickle.dump(all_features_ranked, f)

    return all_features_ranked[:num_ftr]


if __name__ == "__main__":

    start = datetime.now()
    df_list = horz_data_divn('isolet', n_client=60)

    all_features_ranked = federated_mrmr(df_list, 8, 0.2, 'isolet')

    print("Selected feature indices:", all_features_ranked)

    print(f"Time Elapsed: {datetime.now() - start}")
