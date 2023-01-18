import pandas as pd

def preprocessing_data(data_df, dataset_name):
    if dataset_name == 'dc':
        data_df = data_df.rename(columns={'Y': 'Class'})
    if dataset_name == 'nsl':
        data_df = data_df.replace(regex={'anomaly': '1', 'normal': '0'})
        data_df['class'] = data_df['class'].astype(float)
        data_df = data_df.rename(columns={'class': 'Class'})
        data_df.drop(['protocol_type', 'service', 'flag'], axis=1, inplace=True)
        # print(data_df['class'])
    if dataset_name == 'ac':
        data_df.drop(['Time'], axis=1, inplace=True)
    if dataset_name == 'musk':
        data_df.drop(['molecule_name', 'conformation_name'], axis = 1, inplace = True)
    if dataset_name == 'wdbc':
        data_df = data_df.replace(regex={'M': '0', 'B': '1'})
    if dataset_name == 'vowel':
        data_df.drop(['id', 'Train_or_Test'], axis=1, inplace=True)
        data_df = data_df.replace(regex={'Male': '0', 'Female': '1'})
        data_df = data_df.replace(regex={'Andrew': '0', 'Bill': '1', 'David': '2', 'Mark': '3', 'Jo': '4', 'Kate': '5', 
                                         'Penny': '6', 'Rose': '7', 'Mike': '8', 'Nick': '9', 'Rich': '10', 'Tim': '11', 
                                         'Sarah': '12', 'Sue': '13', 'Wendy': '14'})
        data_df = data_df.replace(regex={'hid': 0, 'hEd': 1, 'hAd': 2, 'hYd': 3, 'hOd': 4, 'hUd': 5,
                                         'hId': 0, 'had': 2, 'hod': 4, 'hud': 5, 'hed': 1, 'hyd': 3})
    if dataset_name == 'isolet':
        data_df = data_df.replace(regex={"'1'": 1, "'2'": 2, "'3'": 3, "'4'": 4, "'5'": 5, "'6'": 6, "'7'": 7,
                                         "'8'": 8, "'9'": 9, "'10'": 10, "'11'": 11, "'12'": 12, "'13'": 13,
                                         "'14'": 14, "'15'": 15, "'16'": 16, "'17'": 17, "'18'": 18, "'19'": 19,
                                         "'20'": 20, "'21'": 21, "'22'": 22, "'23'": 23, "'24'": 24, "'25'": 25,
                                         "'26'": 26})
        
    try:
        data_df = data_df.rename(columns={'class': 'Class'})
    except:
        pass
    
    return data_df.astype(float)


def data_part_noniid(data_df1, n_client, degree, curr_dir, file, name):
    for cli in range(0, int(n_client)):
        data_df = data_df1.copy()
        df1 = data_df.pop('Class')
        print("df1_head :", df1.head(3))
        print("client :: ", cli + 1)
        sample_per = float(input("percentage of sample will be present :"))
        file.write("\n sample present in client "+str(cli+1)+":")
        file.write(str(sample_per))
        df = data_df.sample(frac=degree, axis='columns')
        print(len(df.columns))

        df['Class'] = df1
        # print("df = ", df.head())
        print(df.columns)
        df1 = df.loc[df['Class'] == 0]
        df2 = df.loc[df['Class'] == 1]
        df1 = df1.sample(frac=sample_per)
        df2 = df2.sample(frac=sample_per)
        frames = [df1, df2]
        df3 = pd.concat(frames)
        df3 = df3.sample(frac=sample_per).reset_index(drop=True)
        if name == 'nsl':
            df3.to_csv(curr_dir + '/intermediate/nsl-kdd-client-noniid' + str(cli + 1) + '.csv', index=False, header=True)
        else:
            df3.to_csv(curr_dir + '/intermediate/creditcard-client-noniid' + str(cli + 1) + '.csv', index=False,
                       header=True)
        print(len(df3))
        file.write("number of samples in client "+str(n_client)+" : "+str(len(df3)))
        del data_df
        del df1
        del df2
        del df3
    file.write("\n-----------------------------------------------\n")


def data_part_iid(data_df, n_client, curr_dir, name):
    column_names = data_df.columns
    df1 = data_df.loc[data_df['Class'] == 0]
    df2 = data_df.loc[data_df['Class'] == 1]
    base1 = int(len(df1) / n_client)
    base2 = int(len(df2) / n_client)
    print(base1)
    print(base2)
    i = j = 0
    for cli in range(0, int(n_client)):
        df1_cl = pd.DataFrame(columns=column_names)
        print("Data in client number ::", cli)
        print("i= ", i)
        print("j= ", j)
        part1 = base1 + i
        part2 = base2 + j
        print("part1= ", part1)
        print("part2= ", part2)
        for i in range(i, part1):
            df1_cl.loc[df1.index[i]] = df1.iloc[i]
        for j in range(j, part2):
            df1_cl.loc[df2.index[j]] = df2.iloc[j]
        i = i + 1
        j = j + 1
        # df1_cl = df1_cl.sort_values(by=['Time'], ascending = 'False')
        if name == 'nsl':
            df1_cl.to_csv(curr_dir + '/intermediate/nsl-kdd-client-' + str(cli + 1) + '.csv', index=False, header=True)
        else:
            df1_cl.to_csv(curr_dir + '/intermediate/creditcard-client-' + str(cli + 1) + '.csv', index=False, header=True)