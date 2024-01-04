import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy






# Implementation for FedAvg clients

class user_avg():
    def __init__(self,
                 device,
                 id, 
                 train,
                 test,
                 global_model,
                batch_size,
                learning_rate, 
                local_iters
                 ):
        
        self.device = device
        self.local_model = copy.deepcopy(global_model)
        self.id = id  # integer
        self.train_samples = len(train)
        self.test_samples = len(test)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_iters = local_iters
        self.trainloader = DataLoader(train, self.batch_size)
        self.testloader = DataLoader(test, self.batch_size)
        self.testloaderfull = DataLoader(test, self.test_samples)
        self.trainloaderfull = DataLoader(train, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

    
        
        self.loss = nn.NLLLoss()
            

        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate)

     
     
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


    def test(self, global_model):
        self.local_model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return test_acc, loss, y.shape[0]

    def train_error_and_loss(self, global_model):
        self.local_model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            
        return train_acc, loss, y.shape[0]

    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.local_model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.local_model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))



    def train(self):

        for epoch in range(0, self.local_iters):  # local update
            self.local_model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.local_model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
    


        
