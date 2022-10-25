import package1

try:
    from package1.Cluster_kmeans import Cluster_kmeans
    import pandas as pd
    import sys
    import os
    from statistics import mean
    from package1.local_feature_select import local_fs
    from package1.FCMI import FCMI
    from package1.FFMI import FFMI
    from package1.learning_randomforest import learning
    from package1.calc_MI import calc_MI
    from package1.global_feature_select import global_feature_select
    from package1.preprocessing import preprocesing_data
    from package1.learning_knn import knn
    from package1.preprocessing import data_part_noniid
    from package1.preprocessing import data_part_iid
    from package1.global_feature_select import global_feature_select_single
    from package1.federated_forest import fed_tree
except Exception as e:
    print(" Some Modules are missing {}".format(e))


def main():
    print("sys.argv[1]", sys.argv[1])  # output file
    print("sys.argv[2]", sys.argv[2])  # fcmi cluster number
    print("sys.argv[3]", sys.argv[3])  # ffmi cluster number
    print("sys.argv[4]", sys.argv[4])  # dataset name ac / nsl
    print("sys.argv[5]", sys.argv[5])  # dataset type (e.g., iid, noniid)
    print("sys.argv[6]", sys.argv[6])  # number of client
    print("sys.argv[7]", sys.argv[7])  # degree of non-iidness
    # ----Read clients from Database-------
    curr_dir = os.getcwd()
    print(curr_dir)
    f = open(sys.argv[1], "w")
    f.write("\n---------command line arguments-----------\n ")
    f.write("Output file :")
    f.write(sys.argv[1])
    f.write("\n Fcmi cluster number :")
    f.write(sys.argv[2])
    f.write("\n Ffmi cluster number :")
    f.write(sys.argv[3])
    f.write("\n dataset name :")
    f.write(sys.argv[4])
    f.write("\n dataset type :")
    f.write(sys.argv[5])
    f.write("\n number of clients :")
    f.write(sys.argv[6])
    f.write("\n Degree of Non-iidness :")
    f.write(sys.argv[7])
    f.write("\n-----------------------------------------\n ")

    local_feature = []
    feature_list = []
    n_clust_fcmi = int(sys.argv[2])
    n_clust_ffmi = int(sys.argv[3])
    n_client = int(sys.argv[6])
    degree = float(sys.argv[7])
    if sys.argv[4] == 'nsl':
        # data_df = pd.read_csv(curr_dir + "/data/NSL-KDD/KDDTrain+.csv")
        # data_df = preprocesing_data(data_df, sys.argv[4])
        if sys.argv[5] == 'noniid':
            # data_part_noniid(data_df, n_client, degree, curr_dir, f)
            for cli in range(0, n_client):
                data_dfx = pd.read_csv(curr_dir + '/intermediate/nsl-kdd-client-noniid' + str(cli + 1) + '.csv')
                print("cli = ", cli)
                print(data_dfx.columns)
                f.write("\n client :" + str(cli + 1))
                f.write("\n")
                n_clust_ffmi = int(input("input cluster :"))
                f.write("\n affmi cluster:" + str(n_clust_ffmi) + "\n")
                local = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi, f)
                local_feature.append(local)
                print(local)
            # feature_list = global_feature_select(local_feature)
            feature_list = global_feature_select_single(local_feature)
            # converted_list = [str(element) for element in feature_list]
            joined_string = ",".join(feature_list)
            f.write("----Global feature subset----")
            f.write(joined_string)
            print(feature_list)
            f.write("\n")
            for cli in range(0, n_client):
                data_dfx = pd.read_csv(curr_dir + '/intermediate/nsl-kdd-client-noniid' + str(cli + 1) + '.csv')
                print("cli = ", cli)
                # print(data_dfx.columns)
                # print(len(data_dfx.columns))
                df1 = data_dfx.pop('Class')

                f.write("\n----Learning on Global features--------\n" + " Client : " + str(cli + 1) + "----\n")
                data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
                data_dfx['Class'] = df1
                accu = knn(data_dfx, 3)
                print("knn-3:", accu)
                f.write("\n knn-3 :" + str(accu) + "\n")
                accu = knn(data_dfx, 5)
                print("knn-5:", accu)
                f.write("\n knn-5 :" + str(accu) + "\n")
                roc_auc_score = learning(data_dfx)
                f.write("\n roc_auc_score :" + roc_auc_score + "\n")
                print(roc_auc_score)
        elif sys.argv[5] == 'iid':
            # data_part_iid(data_df, n_client, curr_dir)
            for cli in range(0, n_client):
                data_dfx = pd.read_csv(curr_dir + '/intermediate/nsl-kdd-client-' + str(cli + 1) + '.csv')
                print("cli = ", cli)
                f.write("\n----Client : " + str(cli + 1) + "----\n")
                print(data_dfx.columns)
                print(data_dfx.head())
                local = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi, f)
                local_feature.append(local)
                print(local)
            # feature_list = global_feature_select(local_feature)
            feature_list = global_feature_select_single(local_feature)
            joined_string = ",".join(feature_list)

            f.write("\n----Global feature subset----\n")
            f.write(joined_string)
            f.write(joined_string)
            f.write("\n number of global feature subset :" + str(len(feature_list)))
            f.write("\n")
            print(feature_list)
            for cli in range(0, n_client):
                data_dfx = pd.read_csv(curr_dir + '/intermediate/nsl-kdd-client-' + str(cli + 1) + '.csv')
                print("cli = ", cli)
                # print(data_dfx.columns)
                # print(len(data_dfx.columns))
                df1 = data_dfx.pop('Class')

                f.write("\n----Learning on Global features--------\n" + " Client : " + str(cli + 1) + "----\n")
                data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
                data_dfx['Class'] = df1
                accu = knn(data_dfx, 3)
                print("knn-3:", accu)
                f.write("\n knn-3 :" + str(accu) + "\n")
                accu = knn(data_dfx, 5)
                print("knn-5:", accu)
                f.write("\n knn-5 :" + str(accu) + "\n")
                roc_auc_score = learning(data_dfx)
                f.write("\n roc_auc_score :" + roc_auc_score + "\n")
                print(roc_auc_score)

           # acc = package1.federated_forest.fed_tree(10, n_client, 'rf', feature_list)
            # rf = random forest , tree = decision tree , gbdt = gradient boost tree
            # XGBOOST


    elif sys.argv[4] == 'ac':
        data_df = pd.read_csv(curr_dir + "/creditcard.csv")
        data_df = preprocesing_data(data_df, sys.argv[4])
        if sys.argv[5] == 'noniid':
            data_part_noniid(data_df, n_client, degree, curr_dir, f, 'ac')
            for cli in range(0, n_client):
                data_dfx = pd.read_csv(curr_dir + '/intermediate/creditcard-client-noniid' + str(cli + 1) + '.csv')
                print("cli = ", cli)
                print(data_dfx.columns)
                f.write("\n client :" + str(cli + 1))
                f.write("\n")
                n_clust_fcmi = int(input("input cluster fcmi:"))
                f.write("\n fcmi cluster:" + str(n_clust_ffmi) + "\n")
                n_clust_ffmi = int(input("input cluster ffmi:"))
                f.write("\n affmi cluster:" + str(n_clust_ffmi) + "\n")
                local = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi, f)
                local_feature.append(local)
                print(local)
            feature_list = global_feature_select(local_feature)
            # feature_list = global_feature_select_single(local_feature)
            # converted_list = [str(element) for element in feature_list]
            joined_string = ",".join(feature_list)
            f.write("----Global feature subset----")
            f.write(joined_string)
            print(feature_list)
            f.write("\n")
            for cli in range(0, n_client):
                data_dfx = pd.read_csv(curr_dir + '/intermediate/creditcard-client-noniid' + str(cli + 1) + '.csv')
                print("cli = ", cli)
                # print(data_dfx.columns)
                # print(len(data_dfx.columns))
                df1 = data_dfx.pop('Class')

                f.write("\n----Learning on Global features--------\n" + " Client : " + str(cli + 1) + "----\n")
                data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
                data_dfx['Class'] = df1
                accu = knn(data_dfx, 3)
                print("knn-3:", accu)
                f.write("\n knn-3 :" + str(accu) + "\n")
                accu = knn(data_dfx, 5)
                print("knn-5:", accu)
                f.write("\n knn-5 :" + str(accu) + "\n")
                roc_auc_score = learning(data_dfx)
                f.write("\n roc_auc_score :" + roc_auc_score + "\n")
                print(roc_auc_score)
        elif sys.argv[5] == 'iid':
            # data_part_iid(data_df, n_client, curr_dir, 'ac')
            for cli in range(0, n_client):
                data_dfx = pd.read_csv(curr_dir + '/intermediate/creditcard-client-' + str(cli + 1) + '.csv')
                print("cli = ", cli)
                f.write("\n----Client : " + str(cli + 1) + "----\n")
                print(data_dfx.columns)
                print(data_dfx.head())
                n_clust_fcmi = int(input("input cluster fcmi:"))
                f.write("\n fcmi cluster:" + str(n_clust_ffmi) + "\n")
                n_clust_ffmi = int(input("input cluster ffmi:"))
                f.write("\n affmi cluster:" + str(n_clust_ffmi) + "\n")
                local = local_fs(data_dfx, n_clust_fcmi, n_clust_ffmi, f)
                local_feature.append(local)
                print(local)
            #feature_list = global_feature_select(local_feature)
            feature_list = global_feature_select_single(local_feature)
            joined_string = ",".join(feature_list)

            f.write("\n----Global feature subset----\n")
            f.write(joined_string)
            f.write(joined_string)
            f.write("\n number of global feature subset :" + str(len(feature_list)))
            f.write("\n")
            print(feature_list)
            for cli in range(0, n_client):
                data_dfx = pd.read_csv(curr_dir + '/intermediate/creditcard-client-' + str(cli + 1) + '.csv')
                print("cli = ", cli)
                # print(data_dfx.columns)
                # print(len(data_dfx.columns))
                df1 = data_dfx.pop('Class')

                f.write("\n----Learning on Global features--------\n" + " Client : " + str(cli + 1) + "----\n")
                data_dfx = data_dfx[data_dfx.columns.intersection(feature_list)]
                data_dfx['Class'] = df1
                # accu = knn(data_dfx, 3)
                # print("knn-3:", accu)
                # f.write("\n knn-3 :" + str(accu) + "\n")
                # accu = knn(data_dfx, 5)
                # print("knn-5:", accu)
                # f.write("\n knn-5 :" + str(accu) + "\n")
                # roc_auc_score = learning(data_dfx)
                # f.write("\n roc_auc_score :" + roc_auc_score + "\n")
                # print(roc_auc_score)

           # fed_tree(10, n_client, 'rf', feature_list)
            # rf = random forest , tree = decision tree , gbdt = gradient boost tree
            # XGBOOST
    else:
        print("----Wrong Entry----")

    f.close()


if __name__ == "__main__":
    main()
