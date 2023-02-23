#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vert_data_divn import vert_data_divn
from local_feature_select import local_fs
from global_feature_select import global_feature_select, global_feature_select_single
from learning_knn import knn
from learning_randomforest import learning
import os
import pickle

import subprocess

command = ['mpiexec', '-n', '6', 'python', 'VerticalXGBoost.py']

dataset_list = ['ac', 'nsl', 'ionosphere', 'musk', 
                'wdbc', 'vowel', 'wine', 'isolet', 'hillvalley']   
dataset = 'nsl'
lb = 31
ub = 32

lftr = []
max_roc = 0.0
df_list = []

def run_iid(num_ftr, n_client, n_clust_fcmi, n_clust_ffmi, f, dataset):
    
    global lftr
    global df_list
    local_feature = []
    
    if len(lftr) == 0:
        df_list = vert_data_divn(dataset, n_client)
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            f.write("\n----Client : " + str(cli + 1) + "----\n")
            # print(data_dfx.columns)
            # print(data_dfx.head())
            f.write("\n features with this client: " + str(data_dfx.columns))
            f.write("\n fcmi cluster:" + str(n_clust_ffmi) + "\n")
            f.write("\n affmi cluster:" + str(n_clust_ffmi) + "\n")
            local = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi, f)
            local_feature.append(local)
            # print(local)
        lftr = local_feature
    # feature_list = global_feature_select(dataset, local_feature)
    feature_list = global_feature_select_single(lftr, num_ftr)
    joined_string = ",".join(feature_list)

    f.write("\n----Global feature subset----\n")
    f.write(joined_string)
    f.write("\n number of global feature subset :" + str(len(feature_list)))
    f.write("\n")
    # print(feature_list)
    roc = []
    dataframes_to_xgb = []
    for cli in range(0, n_client):
        data_dfx = df_list[cli]
        print("cli = ", cli)
        # print(data_dfx.columns)
        # print(len(data_dfx.columns))
        df1 = data_dfx.iloc[:, -1]

        f.write("\n----Learning on Global features--------\n" + " Client : " + str(cli + 1) + "----\n")
        data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
        data_dfx = data_dfx.assign(Class = df1)
        accu = knn(data_dfx, 3)
        print("knn-3:", accu)
        f.write("\n knn-3 :" + str(accu) + "\n")
        accu = knn(data_dfx, 5)
        print("knn-5:", accu)
        f.write("\n knn-5 :" + str(accu) + "\n")
        ROC_AUC_score = learning(data_dfx, dataset)
        f.write("\n roc_auc_score :" + str(ROC_AUC_score) + "\n")
        roc.append(ROC_AUC_score)      
        dataframes_to_xgb.append(data_dfx)
    with open('dataframes.pkl', 'wb') as f1:
        pickle.dump(dataframes_to_xgb, f1)
    print('Starting XGBoost process...')
    result = subprocess.run(command, stdout=subprocess.PIPE)
    xgb_out = result.stdout.decode('utf-8')
    print(xgb_out)
    print('-------------------------------------')
    print('XGBoost Accuracy: ', xgb_out[-5:])
    roc_avg = sum(roc)/len(roc)
    f.write("\n roc avg: " + str(roc_avg) + "\n")
    f.write("\n XGBoost accuracy: " + xgb_out[-5:])
    return roc_avg

def main(dataset, num_ftr):
    
    global max_roc
    
    dataset_list = ['ac', 'nsl', 'arcene', 'ionosphere', 'relathe', 'musk', 'TOX-171', 
                    'wdbc', 'vowel', 'wine', 'isolet', 'hillvalley']
    FCMI_clust_num = '2'
    FFMI_clust_num = '2'
    dataset = dataset
    dataset_type = 'iid'
    cli_num = '5'
    out_file = 'regvrv_vertical_single_obj_output_'+dataset+'_'+FCMI_clust_num+'_'+FFMI_clust_num+'_iid_'+cli_num+'num_ftr'+str(num_ftr)+'.txt'
    
    f = open(out_file, "w")
    f.write("\n---------command line arguments-----------\n ")
    f.write("Output file :")
    f.write(out_file)
    f.write("\n Fcmi cluster number :")
    f.write(FCMI_clust_num)
    f.write("\n Ffmi cluster number :")
    f.write(FFMI_clust_num)
    f.write("\n dataset name :")
    f.write(dataset)
    f.write("\n dataset type :")
    f.write(dataset_type)
    f.write("\n number of clients :")
    f.write(cli_num)
    f.write("\n-----------------------------------------\n ")

    n_clust_fcmi = int(FCMI_clust_num)
    n_clust_ffmi = int(FFMI_clust_num)
    n_client = int(cli_num)
    
    roc_avg = run_iid(num_ftr, n_client, n_clust_fcmi, n_clust_ffmi, f, dataset)
    
    roc_avg = float(roc_avg)
    if roc_avg > max_roc+0.0001:
        print('roc_avg > max_roc? ', roc_avg > max_roc+0.0001)
        print('roc_avg: ', roc_avg)
        print('max_roc: ', max_roc)
        f.close()
        max_roc = roc_avg
    else:
        f.close()
        os.remove(out_file)

if __name__ == "__main__":
    
    for num_ftr in range(lb, ub):
        print('DATASET NAME: '+dataset)
        main(dataset, num_ftr)