import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np
import random
from tqdm import trange
import json
from sklearn.model_selection import train_test_split
import random
import time
import pandas as pd
import time

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MyDataset(Dataset):
    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)
        print(self.data)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        print(self.data)
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values
        self.len_features = len(self.data.columns) - 1 
        self.num_labels = self.data['Class'].nunique()
        
    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), torch.tensor(self.labels[idx]).float()


def generate_random_numbers(total_sum, num_user):
    print(total_sum)
    ub = total_sum // num_user
    lb = min(num_user, ub)
    
    random_numbers = [random.randint(lb, ub) for _ in range(num_user - 1)]
    
    # Calculate the last number to ensure the sum is total_sum
    last_number = total_sum - sum(random_numbers)
    
    # Shuffle the list to randomize the order
    random.shuffle(random_numbers)
    
    # Add the last number to the list
    random_numbers.append(last_number)
    # print(random_numbers)
    
   
    random.shuffle(random_numbers)
    
    
    return random_numbers



def load_data(csv_file, NUM_USERS):
    random.seed(1)
    np.random.seed(1)
    
    dataset = MyDataset(csv_file)
    # print(len(dataset.features))
    # print(len(dataset.labels))
    unequal_parts = []
    data_div = "random"
    if data_div == "equal":
        n = len(dataset.features)
        m = np.ones(NUM_USERS, dtype=int) * int(n/NUM_USERS)
        print(m)
    else: 
        if NUM_USERS < 2:
            raise ValueError("Size must be greater than 1.")
        else:
            m = generate_random_numbers(len(dataset.features),NUM_USERS)
            print(m)
        

    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    
    lb=0
    for user in trange(NUM_USERS):
        # print("lb :",lb)
        # print("lb+m[user]: ",lb+m[user])
        ub = lb+m[user]
        X[user] += [feature.tolist() for feature in dataset.features[lb:ub]]
        y[user] += [labels.tolist() for labels in dataset.labels[lb:ub]]
        # print(len(X[user]))
        
        lb = lb+m[user]

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    # train_data = {'user_data':{}} 
    # test_data = {'user_data':{}}     

    all_samples=[]

    # Setup 5 users
    # for i in trange(5, ncols=120):
    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.7, stratify=y[i])
        # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])
        num_samples = len(X[i])
        #train_len = int(0.8*num_samples)  #Test 80%
        #test_len = num_samples - train_len
        # print(y_train)
        # input("press")

        train_data['users'].append(uname)
        train_data["user_data"][uname] = {'x': X_train, 'y': y_train}
        train_data['num_samples'].append(len(y_train))
    
        test_data['users'].append(uname)
        test_data["user_data"][uname] = {'x': X_test, 'y': y_test}
        test_data['num_samples'].append(len(y_test))
        # all_samples.append(train_len + test_len)
        # print(train_data)

        # print(test_data)
        # print(test_data)
        # input("press")
    print("~=~=~=~=~=~=~=++++++~=~=~=~=~=~=")
    print("Number of features:", dataset.len_features)
    print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    print("Training_samples:", train_data['num_samples'])
    print("Total training sample:",sum(train_data['num_samples']))
    print("Testing_samples:", test_data['num_samples'])
    print("Total_testing_samples:",sum(test_data['num_samples']))
    print("Median of train samples:", np.median(train_data['num_samples']))
    print("Median of test samples:", np.median(test_data['num_samples']))
    print("~=~=~=~=~=~=~=++++++~=~=~=~=~=~=")
    
    print("Finish Generating Samples")


    return train_data, test_data, dataset.len_features, dataset.num_labels


def read_user_data(id, dataset):
    train_data = dataset[0]['user_data'][id]
    test_data = dataset[1]['user_data'][1]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    
    X_train = torch.Tensor(X_train).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]


    return train_data, test_data