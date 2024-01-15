from MLP_helpers import horz_split, trainModel, fedMLP, acc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
import gc

gc.enable()

tf.random.set_seed(42)


def one_hot_encode(values, num_classes):
    values = values.values.astype(int)
    one_hot = np.zeros((len(values), num_classes))
    one_hot[np.arange(len(values)), values] = 1
    return one_hot


def Fed_MLP(df_list: list, communication_iterations=2):
    """This is the main function for the FedAvg algorithm implemented on the
    trainable MLP(s). It iterated the averaging process for
    {communication_iterations} number of times.

    Parameters
    ----------
    df_list : list
        list of dataframes from each client.
    communication_iterations : TYPE, optional

    Returns
    -------
    accu : float
        accuracy.
    prec : float
        precision.
    rec : float
        recall.
    """
    x_train, y_train, x_test, y_test = horz_split(df_list)

    print("Training on FedMLP....")
    start = datetime.now()

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

    try:
        num_classes = y_train[0].shape[1]
    except:
        num_classes = 1

    fed = None
    for i in range(communication_iterations):
        models = []
        for x, y in tqdm(zip(x_train, y_train), total=len(x_train)):
            models.append(trainModel(x, y, fed))

        fed = fedMLP(models)
        fed.compile(
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

    fed_acc = []
    fed_p = []
    fed_r = []
    print("Evaluating....")
    for i, (x, y) in tqdm(enumerate(zip(x_test, y_test)), total=len(x_test)):
        fed_acc.append(acc(fed, x, y)[0])
        fed_p.append(acc(fed, x, y)[1])
        fed_r.append(acc(fed, x, y)[2])
    accu = sum(fed_acc) / len(fed_acc)
    prec = sum(fed_p) / len(fed_p)
    rec = sum(fed_r) / len(fed_r)

    print(
        "\033[1;33m"
        + f"\nAverage training time per client : {(datetime.now()-start)/len(df_list)}"
        + "\033[0m"
    )

    del fed

    return accu, prec, rec
