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
    def __init__(self,
                 args,
                 device,
                 id, 
                 train_df,
                 val_df,
                 test_df,
                 global_model
                 ):
        
        self.device = device
        self.local_model = copy.deepcopy(global_model)
        self.id = id  # integer
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.local_iters = args.local_iters
        
        self.criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLLLoss

        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate)

       
        train_features, train_labels = self.df_to_tensor(train_df)
        val_features, val_labels = self.df_to_tensor(val_df)
        test_features, test_labels = self.df_to_tensor(test_df)

        # Create datasets
        train_dataset = CustomDataset(train_features, train_labels)
        val_dataset = CustomDataset(val_features, val_labels)
        test_dataset = CustomDataset(test_features, test_labels)

        # Create DataLoaders
        self.trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        self.testloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        self.trainloaderfull = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

       
        self.train_samples = len(train_dataset)
        self.val_samples = len(val_dataset)
        self.test_samples = len(test_dataset)


    # Convert DataFrames to tensors
    def df_to_tensor(self,dataframe):
        features = dataframe.drop(columns='Class').values
        labels = dataframe['Class'].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

     
    def set_parameters(self, model):
        for param, glob_param in zip(self.local_model.parameters(), model.parameters()):
            param.data = glob_param.data.clone()
            
    def get_parameters(self):
        for param in self.local_model.parameters():
            param.detach()
        return self.local_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.local_model.parameters(), new_params):
            param.data = new_param.data.clone()


    def train_error_and_loss(self, global_model):
      
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
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.testloader)
       
        return accuracy, train_loss


    
    def validation(self, global_model):
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
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.testloader)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
                
        return accuracy, validation_loss, precision, recall, f1

    def test(self, global_model):
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
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.testloader)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
                
        return accuracy, validation_loss, precision, recall, f1

    def train(self):

        for epoch in range(0, self.local_iters):  # local update
            self.local_model.train()
            running_loss = 0.0
            
            for features, labels in self.trainloader:
                features, labels = features.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.local_model(features)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() 
    
            # Compute average loss over the epoch
            epoch_loss = running_loss / self.batch_size
            # print(f'Epoch [{epoch+1}/{self.local_iters}], Loss: {epoch_loss:.4f}')


        
