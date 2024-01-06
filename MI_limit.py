from local_feature_select import local_fs, fcmi_and_affmi
import pandas as pd
import os
from preprocessing import preprocessing_data

curr_dir = os.getcwd()

def run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature):
    '''
    Driver function (deprecated) to find upper limits of Mutual Information
    '''
    for cli in range(n_client):
        list_fcmi, avg_Ffmi = fcmi_and_affmi(df_list[cli])
        print('cli: ', cli)
        print(min(list_fcmi))
        print(max(list_fcmi))
        print(min(avg_Ffmi))
        print(max(avg_Ffmi))
    return

dataset_list = ['ionosphere', 'vowel', 'wine', 'hillvalley']
dataset_type = 'iid'
n_clust_fcmi = 2
n_clust_ffmi = 2
f = 0
n_client = 5
local_feature = 0

for dataset in dataset_list:  
    print(dataset)
    if dataset == 'nsl':
        data_df = pd.read_csv(curr_dir + "/datasets/NSL-KDD/KDDTrain+.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        df1 = data_df.iloc[:25000, :]
        df2 = data_df.iloc[25000:50000, :]
        df3 = data_df.iloc[50000:75000, :]
        df4 = data_df.iloc[75000:100000, :]
        df5 = data_df.iloc[100000:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            run_noniid()
        elif dataset_type == 'iid':          
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)

    elif dataset == 'ac':
        data_df = pd.read_csv(curr_dir + "/datasets/annonymized-credit-card/creditcard.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        df1 = data_df.iloc[:57000, :]
        df2 = data_df.iloc[57000:114000, :]
        df3 = data_df.iloc[114000:171000, :]
        df4 = data_df.iloc[171000:228000, :]
        df5 = data_df.iloc[228000:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            run_noniid()
        elif dataset_type == 'iid':
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
            
    elif dataset == 'arcene':
        data_df = pd.read_csv(curr_dir + "/datasets/ARCENE.csv")
        data_df = data_df.sample(frac = 1)
        df1 = data_df.iloc[:40, :]
        df2 = data_df.iloc[40:80, :]
        df3 = data_df.iloc[80:120, :]
        df4 = data_df.iloc[120:160, :]
        df5 = data_df.iloc[160:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            run_noniid()                
        elif dataset_type == 'iid':
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
            
    elif dataset == 'ionosphere':
        data_df = pd.read_csv(curr_dir + "/datasets/ionosphere.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        df1 = data_df.iloc[:70, :]
        df2 = data_df.iloc[70:140, :]
        df3 = data_df.iloc[140:210, :]
        df4 = data_df.iloc[210:280, :]
        df5 = data_df.iloc[280:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
        
    elif dataset == 'relathe':
        data_df = pd.read_csv(curr_dir + "/datasets/RELATHE.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        df1 = data_df.iloc[:300, :]
        df2 = data_df.iloc[300:600, :]
        df3 = data_df.iloc[600:900, :]
        df4 = data_df.iloc[900:1200, :]
        df5 = data_df.iloc[1200:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
    
    elif dataset == 'musk':
        data_df = pd.read_csv(curr_dir + "/datasets/musk_csv.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        df1 = data_df.iloc[:1340, :]
        df2 = data_df.iloc[1340:2680, :]
        df3 = data_df.iloc[2680:4020, :]
        df4 = data_df.iloc[4020:5360, :]
        df5 = data_df.iloc[5360:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
            
    elif dataset == 'TOX-171':
        data_df = pd.read_csv(curr_dir + "/datasets/TOX-171.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        df1 = data_df.iloc[:35, :]
        df2 = data_df.iloc[35:70, :]
        df3 = data_df.iloc[70:105, :]
        df4 = data_df.iloc[105:140, :]
        df5 = data_df.iloc[140:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
            
    elif dataset == 'wdbc':
        data_df = pd.read_csv(curr_dir + "/datasets/WDBC/data.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        df1 = data_df.iloc[:114, :]
        df2 = data_df.iloc[114:228, :]
        df3 = data_df.iloc[228:342, :]
        df4 = data_df.iloc[342:456, :]
        df5 = data_df.iloc[456:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
    
    elif dataset == 'vowel':
        data_df = pd.read_csv(curr_dir + "/datasets/csv_result-dataset_58_vowel.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        data_df = data_df.astype(float)
        df1 = data_df.iloc[:198, :]
        df2 = data_df.iloc[198:396, :]
        df3 = data_df.iloc[396:594, :]
        df4 = data_df.iloc[594:792, :]
        df5 = data_df.iloc[792:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
        
    elif dataset == 'wine':
        data_df = pd.read_csv(curr_dir + "/datasets/wine.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        data_df = data_df.astype(float)
        df1 = data_df.iloc[:36, :]
        df2 = data_df.iloc[36:72, :]
        df3 = data_df.iloc[72:108, :]
        df4 = data_df.iloc[108:144, :]
        df5 = data_df.iloc[144:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
     
    elif dataset == 'isolet':
        data_df = pd.read_csv(curr_dir + "/datasets/isolet_csv.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        data_df = data_df.astype(float)
        df1 = data_df.iloc[:1560, :]
        df2 = data_df.iloc[1560:3120, :]
        df3 = data_df.iloc[3120:4680, :]
        df4 = data_df.iloc[4680:6240, :]
        df5 = data_df.iloc[6240:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
    
    elif dataset == 'hillvalley':
        data_df = pd.read_csv(curr_dir + "/datasets/hill-valley_csv.csv")
        data_df = preprocessing_data(data_df, dataset)
        data_df = data_df.sample(frac = 1)
        data_df = data_df.astype(float)
        df1 = data_df.iloc[:243, :]
        df2 = data_df.iloc[243:486, :]
        df3 = data_df.iloc[486:729, :]
        df4 = data_df.iloc[729:972, :]
        df5 = data_df.iloc[972:, :]
        df_list = [df1, df2, df3, df4, df5]
        if dataset_type == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            run_noniid()
        elif dataset_type == 'iid':          
            # data_part_iid(data_df, n_client, curr_dir)
            run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature)
