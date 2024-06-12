import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def split_data(df):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
    - df (pd.DataFrame): The dataset to be split.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing the training, validation, and test sets.
   """
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(
        train_data, test_size=0.2, random_state=42)
    return train_data, val_data, test_data


def prepare_data(data):
    """
    Prepare the data for training.

    Parameters:
    - data (pd.DataFrame): The dataset to be prepared.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the features (X) and labels (y).
    """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)
    return X, y


def federated_learning(clients_data,
                       num_global_rounds=10,
                       num_local_rounds=20,
                       alpha=1.0,
                       lr=0.01):
    """
    Perform federated learning on the provided client datasets.

    Parameters:
    - clients_data (List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]): List of tuples containing training, validation, and test data for each client.
    - num_global_rounds (int): Number of global training rounds.
    - num_local_rounds (int): Number of local training rounds per client.
    - alpha (float): Regularization strength.
    - lr (float): Learning rate.

    Returns:
    Tuple[Ridge, float, float, float]: Tuple containing the trained global model and evaluation metrics (MSE, RMSE, MAE) on the test set.
    """
    global_model = Ridge(alpha=alpha)

    best_global_model_coefs = None
    best_global_model_intercept = None
    best_rmse = float('inf')
    patience = 50  # Early stopping patience
    no_improvement_rounds = 0

    for global_round in range(num_global_rounds):
        local_model_coefs = []
        local_model_intercepts = []

        for train_data, val_data, test_data in clients_data:
            local_model = Ridge(alpha=alpha)
            X_train, y_train = prepare_data(train_data)

            for local_round in range(num_local_rounds):
                local_model.fit(X_train, y_train)

            local_model_coefs.append(local_model.coef_)
            local_model_intercepts.append(local_model.intercept_)

        global_model_coef = np.mean(local_model_coefs, axis=0)
        global_model_intercept = np.mean(local_model_intercepts, axis=0)
        global_model.coef_ = global_model_coef
        global_model.intercept_ = global_model_intercept

        val_preds = []
        val_true = []

        for train_data, val_data, test_data in clients_data:
            X_val, y_val = prepare_data(val_data)
            outputs = global_model.predict(X_val)
            val_preds.append(outputs)
            val_true.append(y_val)

        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        mse = mean_squared_error(val_true, val_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(val_true, val_preds)

        print(
            f'Global Round {global_round + 1} - Validation MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

        if rmse < best_rmse:
            best_rmse = rmse
            best_global_model_coefs = global_model_coef
            best_global_model_intercept = global_model_intercept
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1

        if no_improvement_rounds >= patience:
            print("Early stopping triggered.")
            break

    global_model.coef_ = best_global_model_coefs
    global_model.intercept_ = best_global_model_intercept
    test_preds = []
    test_true = []

    for train_data, val_data, test_data in clients_data:
        X_test, y_test = prepare_data(test_data)
        outputs = global_model.predict(X_test)
        test_preds.append(outputs)
        test_true.append(y_test)

    test_preds = np.concatenate(test_preds)
    test_true = np.concatenate(test_true)
    mse = mean_squared_error(test_true, test_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_true, test_preds)

    print(f'Test MSE: {mse}, RMSE: {rmse}, MAE: {mae}')
    return global_model, mse, rmse, mae


if __name__ == "__main__":
    for file in os.listdir('./dataframes_to_send/'):
        if 'california' in file or 'boston' in file:
            print('\n', file, '\n')

            file_path = './dataframes_to_send/' + file
            with open(file_path, 'rb') as file:
                clients_data = pickle.load(file)

            split_clients_data = [split_data(df) for df in clients_data]

            best_model, test_mse, test_rmse, test_mae = federated_learning(
                split_clients_data)

            print(test_mse, test_rmse, test_mae)
