import sklearn.feature_selection as skfs
import pandas as pd


def FCMI(data_df1 : pd.DataFrame, 
         f : _io.TextIOWrapper):
    '''
    Wrapper function for computing FCMI scores for features of a given dataframe

    Parameters
    ----------
    data_df1 : pd.DataFrame
        raw dataframe.
    f : _io.TextIOWrapper
        file to be written.

    Returns
    -------
    mi : np.array
        table of mutual information, feature vs. feature.

    '''
    Features = data_df1.iloc[:, :-1]
    # print("Features :", Features.head())
    Target = data_df1.iloc[:, -1:]
    # print("Target :", Target.head())
    mi = skfs.mutual_info_classif(Features,
                                  Target,
                                  discrete_features='auto',
                                  n_neighbors=3, copy=True,
                                  random_state=None)

    print(mi)

    return mi
