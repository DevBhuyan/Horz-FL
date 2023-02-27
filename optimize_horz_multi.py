#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from local_feature_select import local_fs
from global_feature_select import global_feature_select
from learning_knn import knn
from learning_randomforest import learning
import os
from horz_data_divn import horz_data_divn

dataset_list = ['ac', 'nsl', 'ionosphere', 'musk', 
                'wdbc', 'vowel', 'wine', 'isolet', 'hillvalley']   
dataset = 'ac'
lb = 2
ub = 30

lftr = []
df_list = []
max_roc = 0.0

def run_iid(n_client, f, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr):
    
    global lftr
    global df_list
    local_feature = []
    
    if len(lftr) == 0:
        df_list = horz_data_divn(dataset, n_client)
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            f.write("\n----Client : " + str(cli + 1) + "----\n")
            local = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi, f)
            local_feature.append(local)
        lftr = local_feature
    feature_list = global_feature_select(dataset, lftr, num_ftr)
    print('feature list: ', feature_list)
    print('number of features: ', len(feature_list))
    joined_string = ",".join(feature_list)

    f.write("\n----Global feature subset----\n")
    f.write(joined_string)
    f.write("\n number of global feature subset :" + str(len(feature_list)))
    f.write("\n")
    roc = []
    for cli in range(0, n_client):
        data_dfx = df_list[cli].copy(deep = True)
        print("cli = ", cli)
        df_class = data_dfx.pop('Class')

        f.write("\n----Learning on Global features--------\n" + " Client : " + str(cli + 1) + "----\n")
        data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
        data_dfx = data_dfx.assign(Class = df_class)
        accu = knn(data_dfx, 3)
        print("knn-3:", accu)
        f.write("\n knn-3 :" + str(accu) + "\n")
        accu = knn(data_dfx, 5)
        print("knn-5:", accu)
        f.write("\n knn-5 :" + str(accu) + "\n")
        ROC_AUC_score = learning(data_dfx, dataset)
        roc.append(ROC_AUC_score)
        f.write("\n roc_auc_score :" + str(ROC_AUC_score) + "\n")
    roc_avg = sum(roc)/len(roc)
    f.write("\n roc avg: " + str(roc_avg) + "\n")
    
    return roc_avg

def main(dataset, num_ftr):
    
    global max_roc
    
    FCMI_clust_num = '2'
    FFMI_clust_num = '2'
    dataset = dataset
    dataset_type = 'iid'
    cli_num = '5'
    out_file = 'horz_multi_obj_output_'+dataset+'_'+FCMI_clust_num+'_'+FFMI_clust_num+'_iid_'+cli_num+'num_ftr'+str(num_ftr)+'.txt'

    curr_dir = os.getcwd()
    print(curr_dir)
    f = open(out_file, "w")
    f.write("\n---------command line arguments-----------\n ")
    f.write("Output file :")
    f.write(out_file)
    f.write("\n Number of FCMI clusters :")
    f.write(FCMI_clust_num)
    f.write("\n Number of FFMI clusters :")
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
    
    roc_avg = run_iid(n_client, f, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr)
    
    roc_avg = float(roc_avg)
    if roc_avg > max_roc + 0.001:
        print('roc_avg > max_roc? ', roc_avg > max_roc + 0.001)
        print('roc_avg: ', roc_avg)
        print('max_roc: ', max_roc)
        f.close()
        max_roc = roc_avg
    else:
        f.close()
        os.remove(out_file)
    print('max_roc: ', max_roc)
    
if __name__ == "__main__":    
    for num_ftr in range(lb, ub):
        main(dataset, num_ftr)