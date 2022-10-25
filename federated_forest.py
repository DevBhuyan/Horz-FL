from sklearn import tree
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def central_train_trees_in_a_party(trees, args, X_train, y_train, X_test, y_test):
    n_local_models = args.n_teacher_each_partition
    # print("y_test:", y_test)
    # print("x_train:",X_train)
    # print("dataidxs: ", dataidxs)
    # logger.info("In party %d. n_training: %d" % (party_id, len(dataidxs)))
    dataidx_arr = np.arange(len(y_train))
    np.random.shuffle(dataidx_arr)
    # partition the local data to n_local_models parts
    dataidx_each_model = np.array_split(dataidx_arr, n_local_models)
    # print("dataidx_each_model: ", dataidx_each_model)
    for tree_id in range(n_local_models):
        dataid = dataidx_each_model[tree_id]
        # print("dataid:", dataid)
        # logger.info("Training tree %s. n_training: %d" % (str(tree_id), len(dataid)))
        # clf = tree.DecisionTreeClassifier(max_depth=args.max_depth)
        trees[tree_id].fit(X_train[dataid], y_train[dataid])
        acc = trees[tree_id].score(X_test, y_test)
        # logger.info('>> One tree acc: %f' % acc)

    ens_acc = compute_tree_ensemble_accuracy(trees, X_test, y_test)
    logger.info("Local ensemble acc: %f" % ens_acc)
    return trees


def prepare_uniform_weights(n_classes, net_cnt, fill_val=1):
    weights_list = {}

    for net_i in range(net_cnt):
        temp = np.array([fill_val] * n_classes, dtype=np.float32)
        weights_list[net_i] = torch.from_numpy(temp).view(1, -1)

    return weights_list


def normalize_weights(weights):
    Z = np.array([])
    eps = 1e-6
    weights_norm = {}

    for _, weight in weights.items():
        if len(Z) == 0:
            Z = weight.data.numpy()
        else:
            Z = Z + weight.data.numpy()

    for mi, weight in weights.items():
        weights_norm[mi] = weight / torch.from_numpy(Z + eps)

    return weights_norm


def compute_tree_ensemble_accuracy(trees, X_test, y_test):
    y_pred_prob = np.zeros(len(list(y_test)))
    # print("local trees size:", len(trees))
    weights_list = prepare_uniform_weights(2, len(trees))
    weights_norm = normalize_weights(weights_list)
    # print("len of weights norm: ", weights_norm.size())
    # print("weights norm:", weights_norm)
    out_weight = None
    for tree_id, tree in enumerate(trees):
        print(tree)
        pred = trees[tree].predict_proba(X_test)
        # pred (n_samples, n_classes)
        if out_weight is None:
            out_weight = weights_norm[tree_id] * torch.tensor(pred, dtype=torch.float)
        else:
            out_weight += weights_norm[tree_id] * torch.tensor(pred, dtype=torch.float)

    _, pred_label = torch.max(out_weight.data, 1)
    # print("pred label:", pred_label)
    # print("y test:", y_test)
    # print("out weight:", out_weight)
    # print("len of out weight:", len(out_weight))
    correct_num = 0
    # print(pred_label == torch.BoolTensor(y_test))
    correct_num += (pred_label == torch.LongTensor(y_test.values)).sum().item()

    # print("correct num:", correct_num)
    # for i, pred_i in enumerate(out_weight):
    #     pred_class = np.argmax(pred_i)
    #     if pred_class == y_test[i]:
    #         correct_num += 1
    total = len(list(y_test))
    acc = correct_num / total
    return acc


def fed_tree(max_tree_depth, n_local_model, model_type, feature_list, f):
    trees = {tree_i: None for tree_i in range(n_local_model)}
    for tree_i in range(n_local_model):
        if model_type == 'tree':
            trees[tree_i] = tree.DecisionTreeClassifier(max_depth=max_tree_depth)
        elif model_type == 'rf':
            trees[tree_i] = RandomForestClassifier(max_depth=max_tree_depth, n_estimators=100)
        elif model_type == 'gbdt':
            trees[tree_i] = xgb.XGBClassifier(max_depth=max_tree_depth, n_estimators=100,
                                              learning_rate=0.001, gamma=1, reg_lambda=1, tree_method='hist')

    # federated forest
    for tree_id in range(n_local_model):
        # dataid = dataidx_each_model[tree_id]
        # print("dataid:", dataid)
        # logger.info("Training tree %s. n_training: %d" % (str(tree_id), len(dataid)))
        data_df = pd.read_csv(
            '/home/soura/WASP-Phd/project1_fed-fs/intermediate/nsl-kdd-client-noniid' + str(tree_id + 1) + '.csv')
        df1 = data_df.pop('Class')
        data_df = data_df[data_df.columns.intersection(feature_list)]
        data_df['Class'] = df1
        train_df, test_df = train_test_split(data_df, test_size=.2, random_state=1, shuffle=True)
        train_df, valid_df = train_test_split(train_df, test_size=.25, random_state=1, shuffle=True)
        predictors = data_df.columns
        target = predictors[-1]
        print(target)
        predictors = predictors.to_list()
        predictors = predictors[:-1]

        trees[tree_id].fit(train_df[predictors], train_df[target])
        acc = trees[tree_id].score(test_df[predictors], test_df[target])
        logger.info('>> One tree acc: %f' % acc)
        f.write("client " + str(tree_id) + " accuracy" + str(acc) + "\n")
        preds = trees[tree_id].predict(valid_df[predictors])
        print("Roc_auc :", roc_auc_score(valid_df[target].values, preds))

    acc = compute_tree_ensemble_accuracy(trees, test_df[predictors], test_df[target])
    logger.info("Local ensemble acc: %f" % acc)
    f.write("accuracy = " + str(acc) + "\n")


# nsl-kdd99
# single objective
'''feature_list = ['diff_srv_rate',
                'dst_host_diff_srv_rate',
                'dst_host_serror_rate',
                'dst_host_srv_count',
                'dst_host_srv_diff_host_rate',
                'logged_in',
                'num_compromised',
                'num_shells',
                'same_srv_rate',
                'Class']'''

# state of art
'''feature_list = ['dst_host_serror_rate',
                'dst_host_diff_srv_rate',
                'serror_rate', 'logged_in',
                'dst_bytes',
                'dst_host_srv_diff_host_rate',
                'same_srv_rate',
                'rerror_rate',
                'class']'''
# multiobjective
'''feature_list = ['dst_host_serror_rate',
                'is_guest_login',
                'dst_host_diff_srv_rate',
                'num_access_files',
                'hot',
                'dst_bytes',
                'is_host_login',
                # 'dst_host_srv_diff_host_rate',
                # 'land',
                # 'num_compromised',
                # 'logged_in',
                # 'dst_host_srv_count',
                # 'serror_rate'
                'Class']'''
# multi-objective
feature_list = ['dst_bytes',
                'is_host_login',
                'same_srv_rate',
                'root_shell',
                'urgent',
                'num_failed_logins',
                'is_guest_login',
                'diff_srv_rate',
                'num_access_files',
                'Class',
                'logged_in',
                'dst_host_diff_srv_rate',
                'serror_rate',
                'dst_host_srv_diff_host_rate',
                'land',
                'dst_host_serror_rate',
                'num_shells',
                'hot',
                'num_outbound_cmds',
                'num_compromised',
                'rerror_rate',
                'su_attempted',
                'dst_host_srv_count',
                'Class']

# state-of-art
# feature_list = ['V16', 'V21', 'V18', 'Amount', 'V14', 'V17', 'V11', 'Class']
f = open('nsl_non_iid_federated_forest_without_FS.txt', "w")
tree_depth = 9
n_cli = 5
f.write("max tree depth = " + str(tree_depth) + "\n")
f.write("number of clients = " + str(n_cli) + "\n")

fed_tree(tree_depth, n_cli, 'rf', feature_list, f)
