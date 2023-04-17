#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import clone_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# from numpy import exp, log1p
import tensorflow as tf
import gc
gc.enable()

tf.random.set_seed(42)

def one_hot_encode(values, num_classes):
    values = values.values.astype(int)
    one_hot = np.zeros((len(values), num_classes))
    one_hot[np.arange(len(values)), values] = 1
    return one_hot

def horz_split(df_list):
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    for df in df_list:
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        # return x as scaled down and return y as one hot
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train = train.iloc[:, :-1]
        x_test = test.iloc[:, :-1]
        for (_, trcol), (_, tscol) in zip(x_train.iteritems(), x_test.iteritems()):
            try:
                # trcol = exp(trcol)
                trcol = scaler.fit_transform(np.array(trcol).reshape(-1, 1))
                # tscol = exp(tscol)
                tscol = scaler.fit_transform(np.array(tscol).reshape(-1, 1))
            except:
                # trcol = log1p(trcol)
                trcol = scaler.fit_transform(np.array(trcol).reshape(-1, 1))
                # tscol = log1p(tscol)
                tscol = scaler.fit_transform(np.array(tscol).reshape(-1, 1))
        y_train = train.iloc[:, -1]
        y_test = test.iloc[:, -1]
        num_classes = 1 if y_train.nunique()==2 else y_train.nunique()
        
        if num_classes > 1:
            y_train = one_hot_encode(y_train, num_classes)
            y_test = one_hot_encode(y_test, num_classes)
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)
    return x_train_list, y_train_list, x_test_list, y_test_list

def trainModel(x, y, model = None):
    
    callbacks = [ReduceLROnPlateau(monitor = "val_accuracy",
                                 factor = 0.5,
                                 patience = 3,
                                 verbose = 1,
                                 mode = 'max'),
                 EarlyStopping(monitor='val_accuracy',
                               patience=5,           
                               verbose=1,
                               restore_best_weights=True,
                               mode = 'max')]
    
    num_rows = x.shape[0]
    
    xval = x.iloc[int(num_rows*0.8):]
    try:
        yval = y.iloc[int(num_rows*0.8):]
    except:
        yval = y[int(num_rows*0.8):, :]

    input_dim = x.shape[1]
    try:
        num_classes = y.shape[1]
    except:
        num_classes = 1
    
    if model == None:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(units=num_classes, activation='sigmoid' if num_classes==1 else 'softmax')
        ])
    
        model.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy' if num_classes==1 else 'categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'f1_score'])

    model.fit(x, y, callbacks=callbacks, steps_per_epoch=32, epochs=10, validation_data=(xval, yval), verbose=0)

    return model

def fedMLP(models):
    n_models = len(models)
    agg_weights = models[0].get_weights()
    for i in range(1, n_models):
        model_weights = models[i].get_weights()
        for j in range(len(agg_weights)):
            agg_weights[j] = (agg_weights[j] + model_weights[j]) / 2.0
    agg_model = clone_model(models[0])
    agg_model.set_weights(agg_weights)
    return agg_model

def acc(model, x_test, y_test):
    loss, acc, p, r, f = model.evaluate(x_test, y_test)
    return acc, p, r, f
