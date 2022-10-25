from statistics import mean
import sklearn.feature_selection as skfs
import pandas as pd


def FFMI(data_df, f):
    X = list(data_df.columns)
    X = X[:-1]
    print(X)
    data_df = data_df.iloc[:, :-1]
    Avg_Mi = []
    ffmi = []
    for i in range(0, len(X)):
        Target = pd.DataFrame(data_df[X[i]], )
        Features = data_df.loc[:, data_df.columns != X[i]]
        # print(Features.head())
        # print(Target.head())
        mi = skfs.mutual_info_classif(Features,
                                      Target,
                                      discrete_features='auto',
                                      n_neighbors=3, copy=True,
                                      random_state=None)
        print("Doing MI")
        # print(mi)
        ffmi.append([X[i], mi])
        avg_mi = mean(mi)
        Avg_Mi.append(avg_mi)
        # print("Minimum MI:",min_mi)
    print(FFMI)

    f.write("------FFMI-----------\n")
    f.write(str(FFMI))
    f.write("\n")
    return Avg_Mi
