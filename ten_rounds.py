#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from local_feature_select import local_fs
from global_feature_select import global_feature_select, global_feature_select_single
import os
from horz_data_divn import horz_data_divn
from ff import ff
from Fed_MLP import Fed_MLP
import pandas as pd
from normalize import normalize
from sys import exit

dataset_list = [
                # ['ionosphere', 5, 33, 1], 
                # ['musk', 5, 169, 10], 
                # ['wdbc', 5, 31, 1], 
                # ['vowel', 2, 14, 1], 
                ['wine', 5, 13, 1], 
                # ['hillvalley', 5, 100, 5],
                # ['vehicle', 2, 9, 1],
                # ['segmentation', 2, 9, 1],
                # ['ac', 5, 30, 1], 
                # ['nsl', 5, 41, 1], 
                # ['isolet', 80, 617, 80], 
                # ['TOX-171', 500, 5748, 500],
                # ['iot', 5, 28, 1],
                # ['diabetes', 2, 8, 1],
                # ['automobile', 5, 19, 1]
                ]
datasets = pd.DataFrame(dataset_list, columns = ['dataset', 'lb', 'ub', 'step'])

lftr = []
df_list = []
max_MLP = 0.0
obj = 'single'
classifier = 'ff'
ff_list = []
mlp_list = []

def run_iid(n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset):
    
    global lftr
    global df_list
    global obj
    global classifier
    local_feature = []
    
    
    if len(lftr) == 0:
        dlist = []
        df_list = horz_data_divn(dataset, n_client)
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            local, data_df = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi)
            local_feature.append(local)
            dlist.append(data_df)
        dlist = normalize(dlist)
        if classifier == 'ff':
            a, p, r, f = ff(dlist)
        if classifier == 'mlp':
            a, p, r, f = Fed_MLP(dlist)
        lftr = local_feature
        ftrs_returned_by_lfs = max([len(cli) for cli in lftr])
    else:
        a, p, r, f, ftrs_returned_by_lfs = (0, 0, 0, 0, 0)
    
    
    if obj == 'single':
        # Single-Objective ftr sel
        print('SINGLE-OBJECTIVE GLOBAL FTR SELECTION....')
        feature_list, num_avbl_ftrs = global_feature_select_single(lftr, num_ftr)
        # if num_avbl_ftrs < dset['ub']:
        #     print('Number of features supplied by global_feature_select: ', num_avbl_ftrs)
        #     print('Total number of features for dataset : ', dset['ub'])
        #     exit("\n\tShortage of features.... \n\tNot enough features supplied by local_feature_select")
        print('feature list: ', feature_list)
        print('number of features: ', len(feature_list))
    
    
        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class = df1)
            dataframes_to_send.append(data_dfx)
        if classifier == 'ff':
            ff_acc, ff_prec, ff_rec, ff_f1 = ff(dataframes_to_send)
            print(f'ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}, ff_f1: {ff_f1}')
            return 'ff', ff_acc, ff_prec, ff_rec, ff_f1, a, p, r, f, ftrs_returned_by_lfs
        else:
            MLP_acc, MLP_prec, MLP_rec, MLP_f1 = Fed_MLP(dataframes_to_send)
            print(f'MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}, MLP_f1: {MLP_f1}')
            return 'mlp', MLP_acc, MLP_prec, MLP_rec, MLP_f1, a, p, r, f, ftrs_returned_by_lfs
    
    else:
        # Multi-Objective ftr sel
        print('MULTI-OBJECTIVE GLOBAL FTR SELECTION....')
        feature_list, num_avbl_ftrs = global_feature_select(lftr, num_ftr)
        print('feature list: ', feature_list)
        print('number of features: ', len(feature_list))
    
        dataframes_to_send = []
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            df1 = data_dfx.iloc[:, -1]
            data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
            data_dfx = data_dfx.assign(Class = df1)
            dataframes_to_send.append(data_dfx)
        if classifier == 'ff':
            ff_acc, ff_prec, ff_rec, ff_f1 = ff(dataframes_to_send)
            print(f'ff_acc: {ff_acc}, ff_prec: {ff_prec}, ff_rec: {ff_rec}, ff_f1: {ff_f1}')
            return 'ff', ff_acc, ff_prec, ff_rec, ff_f1, a, p, r, f, ftrs_returned_by_lfs
        else:
            MLP_acc, MLP_prec, MLP_rec, MLP_f1 = Fed_MLP(dataframes_to_send)
            print(f'MLP_acc: {MLP_acc}, MLP_prec: {MLP_prec}, MLP_rec: {MLP_rec}, MLP_f1: {MLP_f1}')
            return 'mlp', MLP_acc, MLP_prec, MLP_rec, MLP_f1, a, p, r, f, ftrs_returned_by_lfs
        

def main(dataset, num_ftr, dset, run):
    
    global max_MLP
    global obj
    global classifier
    global ff_list
    global mlp_list
    
    print('Dataset: ', dataset)
    
    FCMI_clust_num = '2'
    FFMI_clust_num = '2'
    dataset = dataset
    cli_num = '5'

    curr_dir = os.getcwd()
    print(curr_dir)

    n_clust_fcmi = int(FCMI_clust_num)
    n_clust_ffmi = int(FFMI_clust_num)
    n_client = int(cli_num)
    
    name, acc, prec, rec, f1, a, p, r, f, ftrs_returned_by_lfs = run_iid(n_client, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset)
      
    
    return name, acc, prec, rec, f1, a, p, r, f, ftrs_returned_by_lfs
    
if __name__ == "__main__":
    
    all_acc = []
    all_prec = []
    all_rec = []
    all_f1 = []
    lfs_accuracies = []
    for _, dset in datasets.iterrows():     # for each dataset
        
        lftr = []
        r = list(range(dset['lb'], dset['ub']+1, dset['step']))
        
        if (dset['ub'])%(dset['step']) != 0:
            r.append(dset['ub'])
        dataset_accuracies = []
        dataset_precs = []
        dataset_recs = []
        dataset_f1 = []
        
        for num_ftr in r:                   # for each number of features
            
            num_ftr_accuracies = []
            num_ftr_precs = []
            num_ftr_recs = []
            num_ftr_f1 = []
            
            for run in range(10):           # for each run
                name, acc, prec, rec, f1, a, p, r, f, ftrs_returned_by_lfs = main(dset['dataset'], num_ftr, dset, run)
                num_ftr_accuracies.append(acc)
                num_ftr_precs.append(prec)
                num_ftr_recs.append(rec)
                num_ftr_f1.append(f1)
                
                if a:
                    lfs_accuracies.append([dset['dataset'], a, p, r, f, ftrs_returned_by_lfs-1])
            
            num_ftr_acc_avg = sum(num_ftr_accuracies)/len(num_ftr_accuracies)
            num_ftr_prec_avg = sum(num_ftr_precs)/len(num_ftr_precs)
            num_ftr_rec_avg = sum(num_ftr_recs)/len(num_ftr_recs)
            num_ftr_f1_avg = sum(num_ftr_f1)/len(num_ftr_f1)
            dataset_accuracies.append(num_ftr_acc_avg)
            dataset_precs.append(num_ftr_prec_avg)
            dataset_recs.append(num_ftr_rec_avg)
            dataset_f1.append(num_ftr_f1_avg)
            print(f'Average accuracy for {num_ftr} features: {num_ftr_acc_avg}')
        all_acc.append(dataset_accuracies)
        all_prec.append(dataset_precs)
        all_rec.append(dataset_recs)
        all_f1.append(dataset_f1)
        
    acc_df = pd.DataFrame(all_acc)
    acc_df.to_pickle('./OUTPUTS/Averaged/'+classifier+'/'+obj+'/horz_output_acc_df.pkl')