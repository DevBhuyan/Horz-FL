from vert_data_divn import vert_data_divn
from local_feature_select import local_fs
from global_feature_select import global_feature_select, global_feature_select_single
from learning_knn import knn
from learning_randomforest import learning

lftr = []

def run_iid(num_ftr, n_client, df_list, n_clust_fcmi, n_clust_ffmi, f, dataset):
    
    global lftr
    local_feature = []
    
    if num_ftr == 30:
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
    out_file = 'vertical_single_obj_output_'+dataset+'_'+FCMI_clust_num+'_'+FFMI_clust_num+'_iid_'+cli_num+'num_ftr'+str(num_ftr)+'.txt'
    
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

    n_clust_fcmi = int(FCMI_clust_num)
    n_clust_ffmi = int(FFMI_clust_num)
    n_client = int(cli_num)
    # degree = float(sys.argv[7])
    
    df_list = vert_data_divn(dataset, n_client)
    roc_avg = run_iid(num_ftr, n_client, df_list, n_clust_fcmi, n_clust_ffmi, f, dataset)

if __name__ == "__main__":
    dataset_list = ['ac', 'nsl', 'ionosphere', 'musk', 
                    'wdbc', 'vowel', 'wine', 'isolet', 'hillvalley']
    i = 'wdbc'     
    for num_ftr in range(30, 31):
        print('DATASET NAME: '+i)
        main(i, num_ftr)