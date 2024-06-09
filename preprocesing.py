from sklearn.preprocessing import LabelEncoder
from SMOTE import smote
import numpy as np
import pandas as pd
import gc

gc.enable()


def label_encode(data_df):
    categorical_cols = data_df.select_dtypes(include=['object']).columns

    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data_df[col] = label_encoders[col].fit_transform(data_df[col])

    return data_df


def preprocessing_data(data_df: pd.DataFrame,
                       dataset_name: str,
                       oversampling: bool = True):

    data_df = label_encode(data_df)

    if dataset_name == "nsl":
        try:
            data_df["class"] = data_df["class"].astype(float)
        except:
            pass
        data_df = data_df.rename(columns={"class": "Class"})
        data_df.drop(["protocol_type", "service", "flag"], axis=1, inplace=True)

    if dataset_name == "ac":
        data_df.drop(["Time"], axis=1, inplace=True)

    if dataset_name == "musk":
        data_df.drop(["molecule_name", "conformation_name"],
                     axis=1, inplace=True)

    if dataset_name == "vowel":
        data_df.drop(["id", "Train_or_Test"], axis=1, inplace=True)
        data_df = label_encode(data_df)

    if dataset_name == "vehicle":
        data_df.drop(
            [
                "Make",
                "Model",
                "Location",
                "Color",
                "Engine",
                "Max Power",
                "Max Torque",
                "Length",
                "Width",
                "Height",
                "Fuel Tank Capacity",
            ],
            axis=1,
            inplace=True,
        )
        data_df = data_df.rename(columns={"Owner": "Class"})
        data_df.dropna(inplace=True)

    if dataset_name == "segmentation":
        data_df.drop(["ID"], axis=1, inplace=True)
        data_df = data_df.rename(columns={"Segmentation": "Class"})
        data_df.dropna(inplace=True)

    if dataset_name == "automobile":

        data_df = data_df.replace(regex={"\?": np.NaN})
        cl = data_df.pop("Class")
        cl += 1
        data_df = data_df.assign(Class=cl)
        data_df = data_df[data_df["Class"] >= 0]
        data_df.dropna(inplace=True)

    if dataset_name == "california":

        data_df['Class'] = data_df['Class']/100000

    if dataset_name == "boston":
        data_df['Class'] = data_df['Class']/10

    try:
        data_df = data_df.rename(columns={"class": "Class"})
    except:
        pass

    if data_df["Class"].min() != 0:
        cl = data_df.pop("Class")
        cl -= 1
        data_df = data_df.assign(Class=cl)

    data_df = data_df.astype(float)
    # BUG: SMOTE may cause issues with certain data, if it does, please comment the next two lines
    if dataset_name not in ["automobile", "california", "boston"] and oversampling:
        data_df = smote(data_df)

    # NORMALIZE
    X = data_df.iloc[:, :-1]
    y = data_df.iloc[:, -1]
    means = X.mean()
    stds = X.std()
    stds[stds == 0] = 1e-6
    X = (X - means) / stds
    data_df = X
    data_df = data_df.assign(Class=y)

    return data_df
