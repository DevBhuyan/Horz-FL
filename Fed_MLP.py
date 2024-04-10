from MLP_helpers import horz_split, trainModel, fedMLP, acc
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import gc

gc.enable()

tf.random.set_seed(42)


def Fed_MLP(df_list: list,
            num_classes: int,
            communication_iterations=2):
    x_train, y_train, x_test, y_test = horz_split(df_list, num_classes)

    print("Training on FedMLP....")
    start = datetime.now()

    fed = None
    for i in range(communication_iterations):
        models = []
        sample_sizes = []
        for x, y, xval, yval in tqdm(zip(x_train, y_train, x_test, y_test), total=len(x_train)):
            models.append(trainModel(x, y, xval, yval, num_classes, fed))
            sample_sizes.append(len(y))

        fed = fedMLP(models, sample_sizes)
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

    print("Evaluating....")
    x_test_concat = pd.concat([pd.DataFrame(x)
                              for x in x_test], axis=0, ignore_index=True)
    y_test_concat = pd.concat([pd.DataFrame(y)
                              for y in y_test], axis=0, ignore_index=True)
    print(x_test_concat, y_test_concat)
    accu, f1 = acc(fed, x_test_concat, y_test_concat)

    print(
        "\033[1;33m"
        + f"\nAverage training time per client : {(datetime.now()-start)/len(df_list)}"
        + "\033[0m"
    )

    del fed

    return accu, f1
