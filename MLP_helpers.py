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

import tensorflow as tf
import gc

gc.enable()

tf.random.set_seed(42)


def one_hot_encode(values, num_classes):
    """One-hot encode categorical values.

    Parameters:
    - values (pandas Series): Categorical values to encode.
    - num_classes (int): Number of classes/categories.

    Returns:
    - one_hot (numpy array): One-hot encoded representation of the values.
    """
    values = values.values.astype(int)
    one_hot = np.zeros((len(values), num_classes))
    one_hot[np.arange(len(values)), values] = 1
    return one_hot


def horz_split(df_list):
    """Horizontally split DataFrames into training and testing sets.

    Parameters:
    - df_list (list): List of DataFrames.

    Returns:
    - x_train_list, y_train_list, x_test_list, y_test_list (lists): Split data.
    """
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    for df in df_list:
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        scaler = MinMaxScaler(feature_range=(0, 1))

        x_train = train.iloc[:, :-1]
        x_test = test.iloc[:, :-1]
        for (_, trcol), (_, tscol) in zip(x_train.iteritems(), x_test.iteritems()):
            trcol = scaler.fit_transform(np.array(trcol).reshape(-1, 1))
            tscol = scaler.fit_transform(np.array(tscol).reshape(-1, 1))

        # One-hot encode labels if more than 2 classes
        y_train = train.iloc[:, -1]
        y_test = test.iloc[:, -1]
        num_classes = 1 if y_train.nunique() == 2 else y_train.nunique()
        if num_classes > 1:
            y_train = one_hot_encode(y_train, num_classes)
            y_test = one_hot_encode(y_test, num_classes)

        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    del df_list
    return x_train_list, y_train_list, x_test_list, y_test_list


def trainModel(x, y, model=None):
    """Train a neural network model.

    Parameters:
    - x (pandas DataFrame): Input features.
    - y (pandas Series or numpy array): Target labels.
    - model (Sequential, optional): Existing model to train or None to create a new one.

    Returns:
    - model (Sequential): Trained neural network model.
    """
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, verbose=1, mode="max"
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            verbose=1,
            restore_best_weights=True,
            mode="max",
        ),
    ]

    num_rows = x.shape[0]
    xval = x.iloc[int(num_rows * 0.8) :]
    yval = y.iloc[int(num_rows * 0.8) :]

    input_dim = x.shape[1]
    num_classes = y.shape[1] if len(y.shape) > 1 else 1

    if model is None:
        model = Sequential(
            [
                Dense(128, activation="relu", input_shape=(input_dim,)),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(32, activation="relu"),
                Dense(
                    units=num_classes,
                    activation="sigmoid" if num_classes == 1 else "softmax",
                ),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=(
                "binary_crossentropy"
                if num_classes == 1
                else "categorical_crossentropy"
            ),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

    model.fit(
        x,
        y,
        callbacks=callbacks,
        steps_per_epoch=32,
        epochs=10,
        validation_data=(xval, yval),
        verbose=0,
    )

    del x
    del y
    del xval
    del yval
    return model


def fedMLP(models):
    """Federate multiple MLP models.

    Parameters:
    - models (list): List of MLP models.

    Returns:
    - agg_model (Sequential): Aggregated MLP model.
    """
    n_models = len(models)
    agg_weights = models[0].get_weights()
    for i in range(1, n_models):
        model_weights = models[i].get_weights()
        for j in range(len(agg_weights)):
            agg_weights[j] = (agg_weights[j] + model_weights[j]) / 2.0
    agg_model = clone_model(models[0])
    agg_model.set_weights(agg_weights)
    del models
    del agg_weights
    return agg_model


def acc(model, x_test, y_test):
    """Evaluate the accuracy, precision, and recall of a model on test data.

    Parameters:
    - model (Sequential): Trained neural network model.
    - x_test (pandas DataFrame): Test features.
    - y_test (pandas Series or numpy array): True test labels.

    Returns:
    - acc (float): Accuracy.
    - p (float): Precision.
    - r (float): Recall.
    """
    loss, acc, p, r = model.evaluate(x_test, y_test)
    return acc, p, r
