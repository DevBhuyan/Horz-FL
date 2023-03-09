def preprocessing_data(data_df, dataset_name):
    if dataset_name == 'nsl':
        data_df = data_df.replace(regex={'anomaly': 1, 'normal': 0})
        data_df['class'] = data_df['class'].astype(float)
        data_df = data_df.rename(columns={'class': 'Class'})
        data_df.drop(['protocol_type', 'service', 'flag'], axis=1, inplace=True)
    if dataset_name == 'ac':
        data_df.drop(['Time'], axis=1, inplace=True)
    if dataset_name == 'musk':
        data_df.drop(['molecule_name', 'conformation_name'], axis = 1, inplace = True)
    if dataset_name == 'wdbc':
        data_df = data_df.replace(regex={'M': 0, 'B': 1})
    if dataset_name == 'vowel':
        data_df.drop(['id', 'Train_or_Test'], axis=1, inplace=True)
        data_df = data_df.replace(regex={'Male': 0, 'Female': 1})
        data_df = data_df.replace(regex={'Andrew': 0, 'Bill': 1, 'David': 2, 'Mark': 3, 'Jo': 4, 'Kate': 5, 
                                         'Penny': 6, 'Rose': 7, 'Mike': 8, 'Nick': 9, 'Rich': 10, 'Tim': 11, 
                                         'Sarah': 12, 'Sue': 13, 'Wendy': 14})
        data_df = data_df.replace(regex={'hid': 0, 'hEd': 1, 'hAd': 2, 'hYd': 3, 'hOd': 4, 'hUd': 5,
                                         'hId': 0, 'had': 2, 'hod': 4, 'hud': 5, 'hed': 1, 'hyd': 3})
    if dataset_name == 'isolet':
        data_df = data_df.replace(regex={"'1'": 1, "'2'": 2, "'3'": 3, "'4'": 4, "'5'": 5, "'6'": 6, "'7'": 7,
                                         "'8'": 8, "'9'": 9, "'10'": 10, "'11'": 11, "'12'": 12, "'13'": 13,
                                         "'14'": 14, "'15'": 15, "'16'": 16, "'17'": 17, "'18'": 18, "'19'": 19,
                                         "'20'": 20, "'21'": 21, "'22'": 22, "'23'": 23, "'24'": 24, "'25'": 25,
                                         "'26'": 26})
    if dataset_name == 'vehicle':
        data_df.drop(['Make', 'Model', 'Location', 'Color', 'Engine', 'Max Power', 'Max Torque', 'Length', 'Width', 'Height', 'Fuel Tank Capacity'], axis = 1, inplace=True)
        data_df = data_df.rename(columns = {'Owner': 'Class'})
        data_df = data_df.replace(regex={'First': 0, 'Second': 1, 'Third': 2})
        data_df = data_df.replace(regex={'Petrol': 1, 'Diesel': 2, 'CNG': 3, 'Electric': 4, 'Hybrid': 5, 'LPG': 6})
        data_df = data_df.replace(regex={'Automatic': 0, 'Manual': 1})
        data_df = data_df.replace(regex={'Corporate': 0, 'Individual': 1, 'Commercial Registration': 2})
        data_df = data_df.replace(regex={'RWD': 0, 'FWD': 1, 'AWD': 2})
        data_df.dropna(inplace = True)
    if dataset_name == 'segmentation':
        data_df.drop(['ID'], axis = 1, inplace=True)
        data_df = data_df.rename(columns = {'Segmentation': 'Class'})
        data_df = data_df.replace(regex={'Male': 0, 'Female': 1, 'Third': 2})
        data_df = data_df.replace(regex={'No': 0, 'Yes': 1})
        data_df = data_df.replace(regex={'Healthcare': 1, 'Engineer': 2, 'Lawyer': 3, 'Entertainment': 4, 'Artist': 5, 'Executive': 6, 'Doctor': 7, 'Homemaker': 8, 'Marketing': 9})
        data_df = data_df.replace(regex={'Low': 0, 'Average': 1, 'High': 2})
        data_df = data_df.replace(regex={'Cat_1': 1, 'Cat_2': 2, 'Cat_3': 3, 'Cat_4': 4, 'Cat_5': 5, 'Cat_6': 6, 'Cat_7': 7})
        data_df = data_df.replace(regex={'A': 1, 'B': 2, 'C': 3, 'D': 4})
        data_df.dropna(inplace = True)
        
    try:
        data_df = data_df.rename(columns={'class': 'Class'})
    except:
        pass
    if data_df['Class'].min() != 0:
        cl = data_df.pop('Class')
        cl -= 1
        data_df = data_df.assign(Class = cl)
    
    return data_df.astype(float)
