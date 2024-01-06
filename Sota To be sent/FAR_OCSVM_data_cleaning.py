import numpy as np
from sklearn.svm import OneClassSVM
from FAR_based_outlier_detection import FAR_based_outlier_detection
import pandas as pd


def compute_rbf_kernel_matrix(data, gamma):

    # Convert data to a numpy array (if it's a pandas DataFrame)
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Compute the pairwise squared Euclidean distances between data points
    pairwise_distances_sq = np.sum((data[:, np.newaxis] - data) ** 2, axis=-1)

    # Apply the RBF kernel formula
    kernel_matrix = np.exp(-gamma * pairwise_distances_sq)

    return kernel_matrix


def compute_linear_kernel_matrix(relevant_features, r):
    # Extract the feature columns and class labels from the relevant_features dataframe
    X = relevant_features.iloc[:, :-1].values
    y = relevant_features.iloc[:, -1].values

    # Compute the kernel matrix using the linear kernel function
    N = len(X)
    K = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # Compute the kernel value for data points X[i] and X[j]
            kernel_value = np.dot(r * X[i], np.transpose(r * X[j]))
            K[i, j] = kernel_value

    return K



def FAR_OCSVM_data_cleaning(df_list, T=0.0, nu=0.9):
    # ... (existing code)

    cleaned_data = []
    for df in df_list:
        # Apply Algorithm 1 to select relevant features
        relevant_features, r, _ = FAR_based_outlier_detection(df, T)

        K = compute_rbf_kernel_matrix(relevant_features, gamma=0.01)

        clf = OneClassSVM(kernel='precomputed', nu=nu)
        clf.fit(K)

        support_vectors = clf.support_vectors_
        coefficients = clf.dual_coef_[0]

        print(support_vectors.shape, coefficients.shape)


    return

        # print(f"Relevant Features shape: {relevant_features.shape}")
        # print(relevant_features.describe())

        # # Apply Algorithm 2 (FAR-OCSVM) to clean the data
        # support_vectors, coefficients = FAR_OCSVM_learning(relevant_features, r, nu)
        # print(f"Support Vectors shape: {support_vectors.shape}")
        # print(f"Coefficients shape: {coefficients.shape}")

        # cleaned_df = relevant_features[decision_function(relevant_features, support_vectors, coefficients, r, 0) > 0]

        # cleaned_data.append(cleaned_df)

    # return cleaned_data