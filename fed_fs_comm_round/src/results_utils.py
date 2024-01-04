import h5py
import numpy as np
import pandas as pd 
import os
import numpy as np
import re
import csv

def convert_csv_to_txt(input_file,output_file):
   
    with open(input_file, 'r') as csv_file, open(output_file, 'w') as space_delimited_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            space_delimited_file.write(' '.join(row) + '\n')

    print(f'CSV file "{input_file}" converted to space-delimited file "{output_file}"')


def read_file(file):
    hf = h5py.File(file, 'r')
    attributes = []
    for key in hf.keys():
        attributes.append(key)
    
    return attributes, hf


def get_data(hf,attributes):
    data = []
    pm = []
    acc_pm = []
    loss_pm = []
    loss_gm = []
    for i in range(len(attributes)):
        ai = hf.get(attributes[i])
        ai = np.array(ai)
        data.append(ai)
    
    return data

def results(path):
    dir_list = os.listdir(path)
    
    no_FS_test_loss = []
    no_FS_test_accuracy = []
    
    fed_mofs_test_loss = []
    fed_mofs_test_accuracy = []
    
    fed_fis_test_loss = []
    fed_fis_test_accuracy = []
    
    anova_test_loss = []
    anova_test_accuracy = []
    
    rfe_test_loss = []
    rfe_test_accuracy = []
    
   
    
   
    
    
    
    for file_name in dir_list:
        if file_name in ['fed_fis.h5', 'fed_mofs.h5', 'no_FS.h5', 'anova.h5', 'rfe.h5']:
            print(file_name)
            attributes, hf = read_file(path+file_name)

            data = get_data(hf,attributes)
            id=0
            for key in hf.keys():
                attributes.append(key)
                # print("id [",id,"] :", key)
                id+=1
                
            
            # gtrl = hf.get('global_train_loss')
            # gtra = hf.get('global_train_accuracy')   
            gtsl = hf.get('global_test_loss')
            gtsa = hf.get('global_test_accuracy')
                
            if file_name == "fed_fis.h5":
                # for perMFL cnn trl is the test accuracy
                
                fed_fis_test_loss.append(np.array(gtsl).tolist())
                fed_fis_test_accuracy.append(np.array(gtsa).tolist())

                
            if file_name == "fed_mofs.h5":
                fed_mofs_test_loss.append(np.array(gtsl).tolist())
                fed_mofs_test_accuracy.append(np.array(gtsa).tolist())
            
            if file_name == "no_FS.h5":
                no_FS_test_loss.append(np.array(gtsl).tolist())
                no_FS_test_accuracy.append(np.array(gtsa).tolist())

            if file_name == "anova.h5":
                anova_test_loss.append(np.array(gtsl).tolist())
                anova_test_accuracy.append(np.array(gtsa).tolist())
            
            if file_name == "rfe.h5":
                rfe_test_loss.append(np.array(gtsl).tolist())
                rfe_test_accuracy.append(np.array(gtsa).tolist())
            

    data_loss = {
        'GR' : np.arange(200),
        'fed_fis' :  fed_fis_test_loss[0][:],
        'fed_mofs' :  fed_mofs_test_loss[0][:],
        'anova' :   anova_test_loss[0][:],
        #'rfe' : rfe_test_loss[0][:],
        'no_fs' : no_FS_test_loss[0][:]
             
                    }

    data_acc = {
        
        'GR' : np.arange(200),
        'fed_fis' :  fed_fis_test_accuracy[0][:],
        'fed_mofs' :  fed_mofs_test_accuracy[0][:],
        'anova' :   anova_test_accuracy[0][:],
        # 'rfe' : rfe_test_accuracy[0][:],
        'no_fs' : no_FS_test_accuracy[0][:]        
    }

    df_l = pd.DataFrame(data_loss)
    df_a = pd.DataFrame(data_acc)

    csv_acc_path = path + "acc.csv"
    csv_loss_path = path + "loss.csv"
    txt_acc_path = path + "acc.txt"
    txt_loss_path = path + "loss.txt"

    df_l.to_csv(csv_loss_path, index=False)
    df_a.to_csv(csv_acc_path, index=False)

    # convert_csv_to_txt(csv_acc_path,txt_acc_path)
    convert_csv_to_txt(csv_loss_path,txt_loss_path)
    

    print(df_l)


# results("/proj/sourasb-220503/fed_fs_communication_round/results/diabetes/")
# input("press")
# results("/proj/sourasb-220503/fed_fs_communication_round/results/hill-valley/")
# input("press")
# results("/proj/sourasb-220503/fed_fs_communication_round/results/ionosphere/")
# input("press")
# results("/proj/sourasb-220503/fed_fs_communication_round/results/KDD99/")
# input("press")
# results("/proj/sourasb-220503/fed_fs_communication_round/results/segmentation/")
# input("press")
# results("/proj/sourasb-220503/fed_fs_communication_round/results/vowel/")
# input("press")
results("/proj/sourasb-220503/fed_fs_communication_round/results/WDBC/")
input("press")