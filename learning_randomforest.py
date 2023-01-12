from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def learning(data_df, dataset):
    all_features = data_df.columns
    target = all_features[-1]

    # print(target)
    all_features = all_features.to_list()
    features = all_features[:-1]
    # features = features.pop()
    # print("features :", features)
    # print("Target :", target)
    TEST_SIZE = 0.25
    VALID_SIZE = 0.25
    RANDOM_STATE = 2018
    RFC_METRIC = 'gini'  # metric used for RandomForestClassifier
    NUM_ESTIMATORS = 100  # number of estimators used for RandomForestClassifier
    NO_JOBS = 4  # number of parallel jobs used for RandomForestClassifier
    MAX_ROUNDS = 1000  # lgb iterations
    EARLY_STOP = 50  # lgb early stop
    OPT_ROUNDS = 1000  # To be adjusted based on best validation rounds
    VERBOSE_EVAL = 50  # Print out metric result
    train_df, valid_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    clf = RandomForestClassifier(n_jobs=NO_JOBS,
                                 random_state=RANDOM_STATE,
                                 criterion=RFC_METRIC,
                                 n_estimators=NUM_ESTIMATORS,
                                 verbose=False)
    clf.fit(train_df[features], train_df[target].values)
    # preds = clf.predict(valid_df[features])
    preds = clf.predict_proba(valid_df[features])
    # print(data_df.shape)
    # print(train_df.shape)
    # print(valid_df.shape)
    # print(preds.shape)
    # print(preds)
    # cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
    # fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
    # sns.heatmap(cm,
    #            xticklabels=['normal', 'anomaly'],
    #            yticklabels=['normal', 'anomaly'],
    #            annot=True, ax=ax1,
    #            linewidths=.2, linecolor="Darkblue", cmap="Blues")
    # plt.title('Confusion Matrix', fontsize=14)
    # plt.show()
    if dataset == 'wine':
        y_true = []
        for value in valid_df[target].values:
            if value == 1.:
                y_true.append([1, 0, 0])
            elif value == 2.:
                y_true.append([0, 1, 0])
            else:
                y_true.append([0, 0, 1])
        y_true = np.asarray(y_true).astype(float)
        # print(y_true)
        ROC_AUC_score = roc_auc_score(y_true, preds, average='macro', max_fpr=1, multi_class='ovo')
    elif dataset == 'vowel':
        y_true = []
        l = 6
        for value in valid_df[target].values:
            ls = []
            for i in range(l):
                if i == int(value):
                    ls.append(1)
                else:
                    ls.append(0)
            y_true.append(ls)
        y_true = np.asarray(y_true).astype(int)
        # print(y_true)
        ROC_AUC_score = roc_auc_score(y_true, preds, average='macro', max_fpr=1, multi_class='ovo')
    elif dataset == 'isolet':
        y_true = []
        l = 26
        for value in valid_df[target].values:
            ls = []
            for i in range(1, l+1):
                if i == int(value):
                    ls.append(1)
                else:
                    ls.append(0)
            y_true.append(ls)
        y_true = np.asarray(y_true).astype(int)
        # print(y_true)
        ROC_AUC_score = roc_auc_score(y_true, preds, average='macro', max_fpr=1, multi_class='ovo')
    else:
        y_true = valid_df[target].values
        preds = clf.predict(valid_df[features])
        pr1int(y_true)
        ROC_AUC_score = roc_auc_score(y_true, preds)
    
    print("Roc_auc :", ROC_AUC_score)
    return ROC_AUC_score
