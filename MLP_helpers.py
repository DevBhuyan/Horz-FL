#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import clone_model
from tensorflow import one_hot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import gc

gc.enable()

tf.random.set_seed(42)


CALLBACKS = [
    ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=3, verbose=1, mode="max"
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        verbose=1,
        restore_best_weights=True,
        mode="max",
    ),
]


def horz_split(df_list, num_classes):
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
    scaler = MinMaxScaler(feature_range=(0, 1))

    for df in df_list:
        train, test = train_test_split(df, test_size=0.2, random_state=42)

        x_train = train.iloc[:, :-1]
        x_test = test.iloc[:, :-1]
        y_train = train.iloc[:, -1]
        y_test = test.iloc[:, -1]
        # OLD CODE FROM V1.0
        # for (_, trcol), (_, tscol) in zip(x_train.iteritems(), x_test.iteritems()):
        #     trcol = scaler.fit_transform(np.array(trcol).reshape(-1, 1))
        #     tscol = scaler.fit_transform(np.array(tscol).reshape(-1, 1))
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # One-hot encode labels if more than 2 classes
        if num_classes > 1:
            y_train = one_hot(y_train, num_classes)
            y_test = one_hot(y_test, num_classes)
            # HINT: One hot encoding is not the problem for low accuracy

        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    return x_train_list, y_train_list, x_test_list, y_test_list


def trainModel(x,
               y,
               xval,
               yval,
               num_classes: int,
               model=None):

    input_dim = x.shape[1]

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
            optimizer=Adam(learning_rate=0.01),
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
        callbacks=CALLBACKS,
        epochs=10,
        validation_data=(xval, yval),
        verbose=0,
    )

    return model


def fedMLP(models, sample_sizes):
    n_models = len(models)
    agg_weights = models[0].get_weights()
    for i in range(1, n_models):
        model_weights = models[i].get_weights()
        for j in range(len(agg_weights)):
            agg_weights[j] = (agg_weights[j] + model_weights[j]) / 2.0
    agg_model = clone_model(models[0])
    agg_model.set_weights(agg_weights)
    return agg_model

    # n_models = len(models)
    # total_samples = sum(sample_sizes)

    # agg_weights = [np.zeros_like(w) for w in models[0].get_weights()]

    # for i in range(n_models):
    #     model_weights = models[i].get_weights()
    #     weight = sample_sizes[i] / total_samples

    #     for j in range(len(agg_weights)):
    #         agg_weights[j] += weight * model_weights[j]

    # agg_model = Sequential(models[0].layers)
    # agg_model.set_weights(agg_weights)

    return agg_model


def acc(model, x_test, y_test):
    loss, acc, p, r = model.evaluate(x_test, y_test)
    if p+r > 0:
        return acc, 2*p*r/(p+r)
    else:
        return acc, 0
