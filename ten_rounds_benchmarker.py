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
import pickle

dataset_list = [
                # ['ionosphere', 5, 33, 1], 
                # ['musk', 5, 169, 10], 
                # ['wdbc', 5, 31, 1], 
                # ['vowel', 2, 14, 1], 
                # ['wine', 5, 13, 1], 
                # ['hillvalley', 5, 100, 5],
                # ['vehicle', 2, 9, 1],
                # ['segmentation', 2, 9, 1],
                # ['ac', 5, 30, 1], 
                # ['nsl', 5, 41, 1], 
                # ['isolet', 80, 617, 80], 
                # ['TOX-171', 500, 5748, 500],
                # ['iot', 5, 28, 1],
                # ['diabetes', 2, 8, 1],
                ['automobile', 5, 19, 1]
                ]
datasets = pd.DataFrame(dataset_list, columns = ['dataset', 'lb', 'ub', 'step'])

lftr = []
df_list = []
max_MLP = 0.0
classifier = 'ff'
ff_list = []
mlp_list = []

def run_iid(n_client, f, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset):
    
    global lftr
    global df_list
    global obj
    global classifier
    
    
    if len(lftr) == 0:
        df_list = horz_data_divn(dataset, n_client, f)
        print(df_list)
    
    if classifier == 'ff':
        ff_acc = ff(df_list)
        f.write("\n federated forest accuracy: " + str(ff_acc) + "\n")
        print('ff_acc: ', ff_acc)
        return 'ff', ff_acc
    else:
        MLP_acc = Fed_MLP(df_list)
        f.write("\n federated MLP accuracy: " + str(MLP_acc) + "\n")
        print('MLP_acc: ', MLP_acc)
        return 'mlp', MLP_acc
    
        

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
    dataset_type = 'iid'
    cli_num = '5'
    out_file = './OUTPUTS/benchmarks/'+classifier+'/horz_output_'+str(run)+'_benchmark_'+dataset+'_'+FCMI_clust_num+'_'+FFMI_clust_num+'_iid_'+cli_num+'num_ftr'+str(num_ftr)+'.txt'

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
    
    name, acc = run_iid(n_client, f, n_clust_fcmi, n_clust_ffmi, dataset, num_ftr, dset)
      
    f.close()
    
    return name, acc
    
if __name__ == "__main__":
    all_acc = []
    for _, dset in datasets.iterrows():     # for each dataset
        lftr = []
        r = dset['ub']
        dataset_accuracies = []
        num_ftr = r
        num_ftr_accuracies = []
        for run in range(10):           # for each run
            name, acc = main(dset['dataset'], num_ftr, dset, run)
            num_ftr_accuracies.append(acc)
        num_ftr_avg = sum(num_ftr_accuracies)/len(num_ftr_accuracies)
        dataset_accuracies.append(num_ftr_avg)
        print(f'Average accuracy for {num_ftr} features: {num_ftr_avg}')
        all_acc.append(dataset_accuracies)
    acc_df = pd.DataFrame(all_acc)
    acc_df.to_pickle('./OUTPUTS/Averaged/'+classifier+'/'+obj+'/horz_output_acc_df.pkl')