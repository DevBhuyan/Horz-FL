#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ff_helpers import horz_split, trainModel, federatedForest
from sklearn.metrics import accuracy_score

# Fetch df_list
# Split train_test divn
# Split X, y
# Train locally
# Aggregate models
# Evaluate

def ff(df_list):
    x_train, y_train, x_test, y_test = horz_split(df_list)
        
    models = []
    for x, y in zip(x_train, y_train):
        models.append(trainModel(x, y))
        
    fed = federatedForest(models)
    
    fed_acc = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        fed_y_pred = fed.predict(x)
        fed_acc.append(accuracy_score(y, fed_y_pred))
        acc = sum(fed_acc)/len(fed_acc)
    return acc