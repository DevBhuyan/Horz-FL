from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def learning(data_df):
    predictors = data_df.columns
    target = predictors[-1]

    # print(target)
    predictors = predictors.to_list()
    predictors = predictors[:-1]
    # predictors = predictors.pop()
    # print("Predictors :", predictors)
    # print("Target :", target)
    TEST_SIZE = 0.25
    VALID_SIZE = 0.25
    RANDOM_STATE = 2018
    RFC_METRIC = 'gini'  # metric used for RandomForrestClassifier
    NUM_ESTIMATORS = 100  # number of estimators used for RandomForrestClassifier
    NO_JOBS = 4  # number of parallel jobs used for RandomForrestClassifier
    MAX_ROUNDS = 1000  # lgb iterations
    EARLY_STOP = 50  # lgb early stop
    OPT_ROUNDS = 1000  # To be adjusted based on best validation rounds
    VERBOSE_EVAL = 50  # Print out metric result
    train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True)
    clf = RandomForestClassifier(n_jobs=NO_JOBS,
                                 random_state=RANDOM_STATE,
                                 criterion=RFC_METRIC,
                                 n_estimators=NUM_ESTIMATORS,
                                 verbose=False)
    clf.fit(train_df[predictors], train_df[target].values)
    preds = clf.predict(valid_df[predictors])
    cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
    # fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
    # sns.heatmap(cm,
    #            xticklabels=['normal', 'anomaly'],
    #            yticklabels=['normal', 'anomaly'],
    #            annot=True, ax=ax1,
    #            linewidths=.2, linecolor="Darkblue", cmap="Blues")
    # plt.title('Confusion Matrix', fontsize=14)
    # plt.show()
    print("Roc_auc :", roc_auc_score(valid_df[target].values, preds))
    return str(roc_auc_score)