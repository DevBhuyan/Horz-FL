#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ff_helpers import horz_split, trainModel, federatedForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    fed_p = []
    fed_r = []
    fed_f = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        fed_y_pred = fed.predict(x)
        fed_acc.append(accuracy_score(y, fed_y_pred))
        fed_p.append(precision_score(y, fed_y_pred, average = 'weighted'))
        fed_r.append(recall_score(y, fed_y_pred, average = 'weighted'))
        fed_f.append(f1_score(y, fed_y_pred, average = 'weighted'))
    ff_acc = sum(fed_acc)/len(fed_acc)
    ff_prec = sum(fed_p)/len(fed_p)
    ff_rec = sum(fed_r)/len(fed_r)
    ff_f1 = sum(fed_f)/len(fed_f)
    
    del(models)
    del(fed)
    return ff_acc, ff_prec, ff_rec, ff_f1