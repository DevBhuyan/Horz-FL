import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn(data_df, k):
    x = data_df.iloc[:, :-1].values
    y = data_df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    learner = KNeighborsClassifier(n_neighbors=k)
    learner.fit(X_train, y_train)
    learner.predict(X_test)[0:5]
    accuracy = learner.score(X_test, y_test)
    # print("knn accuracy :", accuracy)
    return accuracy


