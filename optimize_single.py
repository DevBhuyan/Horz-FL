import pandas as pd
# import sys
import os
# from statistics import mean
from local_feature_select import local_fs
# from FCMI import FCMI
# from FFMI import FFMI
from learning_randomforest import learning
# from calc_MI import calc_MI
from global_feature_select import global_feature_select
from preprocessing import preprocessing_data
from learning_knn import knn
# from preprocessing import data_part_noniid
# from preprocessing import data_part_iid
from global_feature_select import global_feature_select_single

max_roc = 0.0
lftr = []

def run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr):
    global lftr
    # data_part_iid(data_df, n_client, curr_dir, 'ac')
    if num_ftr == 4:
        for cli in range(0, n_client):
            data_dfx = df_list[cli]
            print("cli = ", cli)
            f.write("\n----Client : " + str(cli + 1) + "----\n")
            # print(data_dfx.columns)
            # print(data_dfx.head())
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
    roc_avg = sum(roc)/len(roc)
    f.write("\n roc avg: " + str(roc_avg) + "\n")
    # acc = package1.federated_forest.fed_tree(10, n_client, 'rf', feature_list)
    # rf = random forest , tree = decision tree , gbdt = gradient boost tree
    # XGBOOST
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
    out_file = 'test2_single_obj_output_'+dataset+'_'+FCMI_clust_num+'_'+FFMI_clust_num+'_iid_'+cli_num+'num_ftr'+str(num_ftr)+'.txt'
    
    curr_dir = os.getcwd()
    print(curr_dir)
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
    # f.write("\n Degree of Non-iidness :")
    # f.write(sys.argv[7])
    f.write("\n-----------------------------------------\n ")

    local_feature = []
    n_clust_fcmi = int(FCMI_clust_num)
    n_clust_ffmi = int(FFMI_clust_num)
    n_client = int(cli_num)
    # degree = float(sys.argv[7])
    
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)

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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
            
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
            
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
            
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
    
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
            
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
            
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
    
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
        
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
     
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
    
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
            roc_avg = run_iid(dataset, f, n_client, df_list, n_clust_fcmi, n_clust_ffmi, local_feature, num_ftr)
      
    roc_avg = float(roc_avg)
    if roc_avg > max_roc:
        print('roc_avg > max_roc? ', roc_avg > max_roc)
        print('roc_avg: ', roc_avg)
        print('max_roc: ', max_roc)
        f.close()
        max_roc = roc_avg
    else:
        f.close()
        os.remove(out_file)
    


if __name__ == "__main__":
    dataset_list = ['ac', 'nsl', 'ionosphere', 'musk', 
                    'wdbc', 'vowel', 'wine', 'isolet', 'hillvalley']
    i = 'relathe'     
    for num_ftr in range(4, 4000):
        print('DATASET NAME: '+i)
        main(i, num_ftr)
