import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from src.data_prep import CustomDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Implementation for FedAvg clients

class user_avg():
    """
    Class representing a user in the Federated Averaging (FedAvg) setup.

    Attributes:
    - device (torch.device): Device (CPU or GPU) on which the model operates.
    - local_model (nn.Module): Local copy of the global model for this user.
    - id (int): Identifier for the user.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for training.
    - local_iters (int): Number of local training iterations.
    - criterion (torch.nn.CrossEntropyLoss): Loss function for training.
    - optimizer (torch.optim.SGD): Optimizer used for training.
    - trainloader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - testloader (DataLoader): DataLoader for test data.
    - trainloaderfull (DataLoader): DataLoader for full training data (used for evaluation).
    - train_samples (int): Number of samples in the training set.
    - val_samples (int): Number of samples in the validation set.
    - test_samples (int): Number of samples in the test set.

    Methods:
    - df_to_tensor(dataframe): Converts a pandas DataFrame to PyTorch tensors.
    - set_parameters(model): Sets parameters of the local model to those of a given global model.
    - get_parameters(): Retrieves parameters of the local model.
    - clone_model_paramenter(param, clone_param): Clones parameters of the local model.
    - get_updated_parameters(): Retrieves updated parameters after training.
    - update_parameters(new_params): Updates parameters of the local model with new parameters.
    - train_error_and_loss(global_model): Computes training accuracy and loss.
    - validation(global_model): Performs validation and computes metrics.
    - test(global_model): Performs test and computes metrics.
    - train(): Performs local training iterations.

    """

    def __init__(self,
                 args,
                 device,
                 id,
                 train_df,
                 val_df,
                 test_df,
                 global_model
                 ):
        """
        Initialize a user in the FedAvg setup.

        Parameters:
        - args (object): Object containing arguments such as batch_size, lr, and local_iters.
        - device (torch.device): Device (CPU or GPU) on which the model operates.
        - id (int): Identifier for the user.
        - train_df (pd.DataFrame): DataFrame containing training data.
        - val_df (pd.DataFrame): DataFrame containing validation data.
        - test_df (pd.DataFrame): DataFrame containing test data.
        - global_model (nn.Module): Global model shared among users.
        """

        self.device = device
        self.local_model = copy.deepcopy(global_model)
        self.id = id  # integer
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.local_iters = args.local_iters

        self.criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLLLoss

        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(), lr=self.learning_rate)

        train_features, train_labels = self.df_to_tensor(train_df)
        val_features, val_labels = self.df_to_tensor(val_df)
        test_features, test_labels = self.df_to_tensor(test_df)

        # Create datasets
        train_dataset = CustomDataset(train_features, train_labels)
        val_dataset = CustomDataset(val_features, val_labels)
        test_dataset = CustomDataset(test_features, test_labels)

        # Create DataLoaders
        self.trainloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=len(val_dataset), shuffle=False)
        self.testloader = DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False)

        self.trainloaderfull = DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=True)

        self.train_samples = len(train_dataset)
        self.val_samples = len(val_dataset)
        self.test_samples = len(test_dataset)

    # Convert DataFrames to tensors
    def df_to_tensor(self, dataframe):
        """
        Convert a pandas DataFrame to PyTorch tensors.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing features and labels.

        Returns:
        - torch.tensor: Tensor of features.
        - torch.tensor: Tensor of labels.
        """
        features = dataframe.drop(columns='Class').values
        labels = dataframe['Class'].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def set_parameters(self, model):
        """
        Set parameters of the local model to those of a given global model.

        Parameters:
        - model (nn.Module): Global model whose parameters will be copied.
        """
        for param, glob_param in zip(self.local_model.parameters(), model.parameters()):
            param.data = glob_param.data.clone()

    def get_parameters(self):
        """
        Retrieve parameters of the local model.

        Returns:
        - torch.parameters: Parameters of the local model.
        """
        for param in self.local_model.parameters():
            param.detach()
        return self.local_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        """
        Clone parameters of the local model.

        Parameters:
        - param (torch.parameters): Parameters of the local model.
        - clone_param (torch.parameters): Cloned parameters of the local model.
        """
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        """
        Retrieve updated parameters after training.

        Returns:
        - torch.parameters: Updated parameters of the local model.
        """
        return self.local_weight_updated

    def update_parameters(self, new_params):
        """
        Update parameters of the local model with new parameters.

        Parameters:
        - new_params (torch.parameters): New parameters to update.
        """
        for param, new_param in zip(self.local_model.parameters(), new_params):
            param.data = new_param.data.clone()

    def train_error_and_loss(self, global_model):
        """
        Compute training accuracy and loss.

        Parameters:
        - global_model (nn.Module): Global model for comparison.

        Returns:
        - float: Accuracy of the local model on the training set.
        - float: Loss of the local model on the training set.
        """

        # Set the model to evaluation mode
        self.local_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []

        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.trainloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                # Collect predicted labels
                y_pred.extend(predicted.cpu().numpy())

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.testloader)

        return accuracy, train_loss

    def validation(self, global_model):
        """
        Perform validation and compute metrics.

        Parameters:
        - global_model (nn.Module): Global model for comparison.

        Returns:
        - float: Accuracy of the local model on the validation set.
        - float: Loss of the local model on the validation set.
        - float: Precision of the local model on the validation set.
        - float: Recall of the local model on the validation set.
        - float: F1 score of the local model on the validation set.
        """
        # Set the model to evaluation mode
        self.local_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []

        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                # Collect predicted labels
                y_pred.extend(predicted.cpu().numpy())

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.testloader)
        # Use 'macro' for unweighted
        precision = precision_score(
            y_true, y_pred, average='weighted', zero_division=0)
        # Use 'macro' for unweighted
        recall = recall_score(
            y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted',
                      zero_division=0)  # Use 'macro' for unweighted

        return accuracy, validation_loss, precision, recall, f1

    def test(self, global_model):
        """
        Perform test and compute metrics.

        Parameters:
        - global_model (nn.Module): Global model for comparison.

        Returns:
        - float: Accuracy of the local model on the test set.
        - float: Loss of the local model on the test set.
        - float: Precision of the local model on the test set.
        - float: Recall of the local model on the test set.
        - float: F1 score of the local model on the test set.
        """
        # Set the model to evaluation mode
        self.local_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []

        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                # Collect predicted labels
                y_pred.extend(predicted.cpu().numpy())

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.testloader)
        # Use 'macro' for unweighted
        precision = precision_score(
            y_true, y_pred, average='weighted', zero_division=0)
        # Use 'macro' for unweighted
        recall = recall_score(
            y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted',
                      zero_division=0)  # Use 'macro' for unweighted

        return accuracy, validation_loss, precision, recall, f1

    def train(self):

        for epoch in range(0, self.local_iters):  # local update
            self.local_model.train()
            running_loss = 0.0

            for features, labels in self.trainloader:
                features, labels = features.to(
                    self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.local_model(features)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Compute average loss over the epoch
            epoch_loss = running_loss / self.batch_size
            # print(f'Epoch [{epoch+1}/{self.local_iters}], Loss: {epoch_loss:.4f}')
