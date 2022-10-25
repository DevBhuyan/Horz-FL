import pandas as pd


def preprocesing_data(data_df, dataset_name):
    if dataset_name == 'dc':
        data_df = data_df.rename(columns={'Y': 'Class'})
    if dataset_name == 'nsl':
        data_df = data_df.replace(regex={'anomaly': '1', 'normal': '0'})
        data_df['class'] = data_df['class'].astype(float)
        data_df = data_df.rename(columns={'class': 'Class'})
        data_df.drop(['protocol_type', 'service', 'flag'], axis=1, inplace=True)
        # print(data_df['class'])
    if dataset_name == 'ac':
        data_df.drop(['Time', ], axis=1, inplace=True)

    return data_df


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