#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from local_feature_select import local_fs
from global_feature_select import global_feature_select, global_feature_select_single
import os
from horz_data_divn import horz_data_divn
from ff import ff
from Fed_MLP import Fed_MLP
import pandas as pd
from sys import exit

dataset_list = [
                # ['ac', 5, 30, 5], 
                # ['nsl', 5, 41, 5], 
                # ['ionosphere', 5, 33, 5], 
                # ['musk', 5, 169, 20], 
                ['wdbc', 5, 31, 5], 
                # ['vowel', 5, 14, 2], 
                ['wine', 5, 13, 2], 
                # ['isolet', 80, 617, 80], 
                # ['hillvalley', 15, 100, 15],
                # ['vehicle', 4, 9, 2],
                ['segmentation', 4, 9, 2]
                ]
datasets = pd.DataFrame(dataset_list, columns = ['dataset', 'lb', 'ub', 'step'])

lftr = []
df_list = []
max_MLP = 0.0

def run_iid(n_client, f, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset):
    
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
    
    
    # Single-Objective ftr sel
    print('SINGLE-OBJECTIVE GLOBAL FTR SELECTION....')
    feature_list, num_avbl_ftrs = global_feature_select_single(lftr, num_ftr)
    if num_avbl_ftrs < dset['ub']:
        print('num_avbl_ftrs: ', num_avbl_ftrs)
        print('dset[ub]: ', dset['ub'])
        exit("Shortage of features.... \n Not enough features supplied by local_feature_select")
    print('feature list: ', feature_list)
    print('number of features: ', len(feature_list))
    joined_string = ",".join(feature_list)

    f.write("\n----Single-objective global feature subset----\n")
    f.write(joined_string)
    f.write("\n number of global feature subset :" + str(len(feature_list)))
    f.write("\n")
    dataframes_to_send = []
    f.write("\n----Learning on Global features--------\n")
    for cli in range(0, n_client):
        data_dfx = df_list[cli]
        print("cli = ", cli)
        df1 = data_dfx.iloc[:, -1]
        data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
        data_dfx = data_dfx.assign(Class = df1)
        dataframes_to_send.append(data_dfx)
    ff_acc = ff(dataframes_to_send)
    MLP_acc = Fed_MLP(dataframes_to_send)
    f.write("\n federated forest accuracy: " + str(ff_acc) + "\n")
    f.write("\n federated MLP accuracy: " + str(MLP_acc) + "\n")
    print('ff_acc: ', ff_acc)
    print('MLP_acc: ', MLP_acc)
    
    
    # Multi-Objective ftr sel
    print('MULTI-OBJECTIVE GLOBAL FTR SELECTION....')
    feature_list = global_feature_select(lftr, num_ftr)
    print('feature list: ', feature_list)
    print('number of features: ', len(feature_list))
    joined_string = ",".join(feature_list)

    f.write("\n----Multi-objective global feature subset----\n")
    f.write(joined_string)
    f.write("\n number of global feature subset :" + str(len(feature_list)))
    f.write("\n")
    dataframes_to_send = []
    f.write("\n----Learning on Global features--------\n")
    for cli in range(0, n_client):
        data_dfx = df_list[cli]
        print("cli = ", cli)
        df1 = data_dfx.iloc[:, -1]
        data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
        data_dfx = data_dfx.assign(Class = df1)
        dataframes_to_send.append(data_dfx)
    ff_acc = ff(dataframes_to_send)
    MLP_acc = Fed_MLP(dataframes_to_send)
    f.write("\n federated forest accuracy: " + str(ff_acc) + "\n")
    f.write("\n federated MLP accuracy: " + str(MLP_acc) + "\n")
    print('ff_acc: ', ff_acc)
    print('MLP_acc: ', MLP_acc)
    
    return

def main(dataset, num_ftr, dset):
    
    global max_MLP
    
    print('Dataset: ', dataset)
    
    FCMI_clust_num = '2'
    FFMI_clust_num = '2'
    dataset = dataset
    dataset_type = 'iid'
    cli_num = '5'
    out_file = 'plot_of_horz_output_'+dataset+'_'+FCMI_clust_num+'_'+FFMI_clust_num+'_iid_'+cli_num+'num_ftr'+str(num_ftr)+'.txt'

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
    
    run_iid(n_client, f, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset)
    
    f.close()
    
if __name__ == "__main__":
    for _, dset in datasets.iterrows():
        lftr = []
        r = list(range(dset['lb'], dset['ub']+1, dset['step']))
        if (dset['ub'])%5 != 0:
            r.append(dset['ub'])
        for num_ftr in r:
            main(dset['dataset'], num_ftr, dset)