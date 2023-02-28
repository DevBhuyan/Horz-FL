#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import math

def horz_split(df_list):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for df in df_list:
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        x_train.append(train.iloc[:, :-1])
        y_train.append(train.iloc[:, -1])
        x_test.append(test.iloc[:, :-1])
        y_test.append(test.iloc[:, -1])
    return x_train, y_train, x_test, y_test

def trainModel(x, y):
    clf = RandomForestClassifier(n_estimators=100,
                                 max_depth=20,
                                 random_state=42)
    clf.fit(x, y)
    return clf

def aggregateForests(agg, model, n):
    # Aggregate the estimators
    size = model.n_estimators
    dupl = math.floor(n/size)
    for i in range(dupl):
        agg.estimators_ += model.estimators_
        agg.n_estimators = len(agg.estimators_)
        
    return agg

def getMaxForest(model_list):
    # normalize over the maximum number of trees so each model is weighted the same in the aggregated model
    sizes = [len(model.estimators_) for model in model_list]
    
    return max(sizes)

def federatedForest(model_list):
    # Aggregate all models and normalize
    ff = model_list[0]
    max_size = getMaxForest(model_list)
    for i, model in enumerate(model_list):
        if i > 0:
            ff = aggregateForests(ff, model_list[i], max_size)
            
    return ff
