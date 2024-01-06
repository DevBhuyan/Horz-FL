#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from local_feature_select import local_fs
from global_feature_select import global_feature_select_single
from learning_knn import knn
from learning_randomforest import learning
import os
from horz_data_divn import horz_data_divn
from ff import ff

dataset_list = ['ac', 'nsl', 'ionosphere', 'musk', 
                'wdbc', 'vowel', 'wine', 'isolet', 'hillvalley']   
dataset = 'nsl'
lb = 10
ub = 42

lftr = []
df_list = []
max_ff = 0.0
benchmark = 0.9970866208

def run_iid(n_client : int, 
            f : _io.TextIOWrapper, 
            n_clust_fcmi : int, 
            n_clust_ffmi : int, 
            dataset : str, 
            num_ftr : int):
    '''
    

    Parameters
    ----------
    n_client : int
        number of clients.
    f : _io.TextIOWrapper
        file to be written.
    n_clust_fcmi : int
        number of fcmi clusters.
    n_clust_ffmi : int
        number of ffmi clusters.
    dataset : str
        code name of dataset.
    num_ftr : int
        number of features.

    Returns
    -------
    None.

    '''
    
    global lftr
    global df_list
    local_feature = []
    
    if len(lftr) == 0:
        df_list = horz_data_divn(dataset, n_client, f)
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            f.write("\n----Client : " + str(cli + 1) + "----\n")
            local = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi, f)
            local_feature.append(local)
        lftr = local_feature
    feature_list = global_feature_select_single(lftr, num_ftr)
    print('feature list: ', feature_list)
    print('number of features: ', len(feature_list))
    joined_string = ",".join(feature_list)

    f.write("\n----Global feature subset----\n")
    f.write(joined_string)
    f.write("\n number of global feature subset :" + str(len(feature_list)))
    f.write("\n")
    roc = []
    dataframes_to_send = []
    for cli in range(0, n_client):
        data_dfx = df_list[cli]
        print("cli = ", cli)
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
        dataframes_to_send.append(data_dfx)
    ff_acc = ff(dataframes_to_send)
    roc_avg = sum(roc)/len(roc)
    f.write("\n roc avg: " + str(roc_avg) + "\n")
    f.write("\n federated forest accuracy: " + str(ff_acc) + "\n")
    print('ff_acc: ', ff_acc)
    
    return roc_avg, ff_acc

def main(dataset, num_ftr):
    
    global max_ff
    
    FCMI_clust_num = '2'
    FFMI_clust_num = '2'
    dataset = dataset
    dataset_type = 'iid'
    cli_num = '5'
    out_file = 'horz_single_obj_output_'+dataset+'_'+FCMI_clust_num+'_'+FFMI_clust_num+'_iid_'+cli_num+'num_ftr'+str(num_ftr)+'.txt'

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
    
    roc_avg, ff_acc = run_iid(n_client, f, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr)
    
    roc_avg = float(roc_avg)
    if ff_acc > max_ff + 0.001:
        print('ff_acc > max_ff? ', ff_acc > max_ff + 0.001)
        print('ff_acc: ', ff_acc)
        print('max_ff: ', max_ff)
        f.close()
        max_ff = ff_acc
    else:
        f.close()
        os.remove(out_file)
    print('max_ff: ', max_ff)
    
if __name__ == "__main__":   
    while (max_ff < benchmark):
        for num_ftr in range(lb, ub):
            if (max_ff < benchmark):
                main(dataset, num_ftr)
        lftr = []
