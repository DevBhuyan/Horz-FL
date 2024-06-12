
from src.local_update import user_avg
from src.train_model import DNN
import torch
import os
import h5py
import numpy as np
import copy
from datetime import date
from tqdm import tqdm, trange
import pickle
import sys
from sklearn.model_selection import train_test_split
import wandb
import datetime
import sys
from torchmetrics import Precision, Recall, F1Score
import statistics
import pickle


class server_avg():
    def __init__(self,
                args,
                device,
                current_directory,
                times):
        self.times=times
        self.current_directory=current_directory   
        self.device=device
        self.dataset_name=args.dataset_name
        self.num_glob_iters=args.global_iters
        self.local_iters=args.local_iters
        self.batch_size=args.batch_size
        self.learning_rate=args.lr
        self.total_train_samples=0
        self.num_users=args.tot_users*args.p
        self.fs_method=args.fs_method
        self.noniidness=args.non_iidness
        self.n_classes=args.n_classes
        self.only_test=args.only_test

        self.users = []
        self.selected_users = []

        self.global_train_acc = []
        self.global_train_loss = [] 
        self.global_val_acc = [] 
        self.global_val_loss = []
        self.global_precision = []
        self.global_recall = []
        self.global_f1score = []

        self.minimum_val_loss = 1000000000000.0

        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb = wandb.init(project="FedMOFS", name="FedMOFS_%s_%d" % (date_and_time, self.num_users), mode=None if args.wandb else "disabled")


        pickle_file_path = current_directory + '/to_banerjee_sir/new_dataset/' + 'df_list_for_%s_%d_%d_%.1f_%s.pkl' % (self.fs_method, args.tot_users, args.num_features, self.noniidness, self.dataset_name)

        with open(pickle_file_path, 'rb') as file:
        # Load the contents of the file into a Python object
            datasets = pickle.load(file)

        self.features = datasets[0].columns.tolist()
        self.num_features = len(self.features)-1
        print(self.features)
        print(datasets[0])
        self.mid_dim=100
        self.global_model = DNN(input_dim=self.num_features, mid_dim=self.mid_dim, output_dim=self.n_classes).to(device)
        
        for i in trange(args.tot_users):
            user_id = i
            # train, test = read_user_data(user_id, datasets[i])
            train_df, temp_df = train_test_split(datasets[i], test_size=0.4, random_state=42+times)

            # Then, split the temp_df into validation and test sets (50% of temp_df each, which is 10% of the original data)
            val_df, test_df = train_test_split(temp_df, test_size=0.4, random_state=42+times)

            # Print the sizes of the splits to verify
            print(f'total set size: {datasets[i].shape[0]}')
            print(f'Training set size: {train_df.shape[0]}')
            print(f'Validation set size: {val_df.shape[0]}')
            print(f'Test set size: {test_df.shape[0]}')

            
            
            user = user_avg(args,
                            device,
                            id, 
                            train_df, 
                            val_df, 
                            test_df, 
                            self.global_model
                            )
           
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Finished creating FedAvg server.")

        # Convert DataFrames to tensors
        def df_to_tensor(dataframe):
            features = dataframe.drop(columns='label').values
            labels = dataframe['label'].values
            return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

        

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model)

    def add_parameters(self, user, ratio):
        model = self.global_model.parameters()
        for server_param, user_param in zip(self.global_model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)


    def select_users(self, round, subset_users):
        
        if int(subset_users) == len(self.users):
            return self.users
        elif  subset_users < len(self.users):
            np.random.seed(round)
            print(subset_users)
            return np.random.choice(self.users, int(subset_users), replace=False)
            
        else: 

            assert (int(subset_users) > len(self.users))
            
    def save_model(self, glob_iter):
        if glob_iter == self.num_glob_iters-1:
            model_path = self.current_directory + "/models/" + self.fs_method + "/global_model/" 
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_val_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "%s_noniid_%.1f_user_%d_features_%d_server_checkpoint_GR_%d.pt" %(self.dataset_name, self.noniidness, self.num_users, self.num_features, glob_iter)))
            
        if self.global_val_loss[glob_iter] < self.minimum_val_loss:
            self.minimum_val_loss = self.global_val_loss[glob_iter]
            model_path = self.current_directory + "/models/" + self.fs_method + "/global_model/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            print(f"global model : {self.global_model}")
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_val_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "%s_noniid_%.1f_user_%d_features_%d_best_checkpoint.pt" %(self.dataset_name, self.noniidness, self.num_users, self.num_features)))

    def test_error_and_loss(self, model):
        accs = []
        losses = []
        precisions = []
        recalls = []
        f1s = []
        for c in self.users:
            accuracy, loss, precision, recall, f1 = c.test(model.parameters())
            accs.append(accuracy)
            losses.append(loss)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return accs, losses, precisions, recalls, f1s
        

    
    def test_best_model(self):
        model = DNN(input_dim=self.num_features, mid_dim=self.mid_dim, output_dim=self.n_classes).to(self.device)

        model_path = self.current_directory + "/models/" + self.fs_method + "/global_model/" + "%s_noniid_%.1f_user_%d_features_%d_best_checkpoint.pt" %(self.dataset_name, self.noniidness, self.num_users, self.num_features)
        print(model_path)
        print(f"model : {model}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        accs, losses, precisions, recalls, f1s = self.test_error_and_loss(model)
        
        global_test_acc = statistics.mean(accs)
        global_test_loss = statistics.mean(losses)
        global_test_precision = statistics.mean(precisions)
        global_test_recall = statistics.mean(recalls)
        global_test_f1score = statistics.mean(f1s)
        


        print(f"Global test accurancy: {global_test_acc}")
        print(f"Global test_loss: {global_test_loss}")
        print(f"Global test Precision: {global_test_precision}")
        print(f"Global test Recall: {global_test_recall}")
        print(f"Global test f1score: {global_test_f1score}")

        print("test metric %.2f %.2f %.2f %.2f %.2f" %(global_test_acc, global_test_loss, global_test_precision, global_test_recall, global_test_f1score ))
        print("%.2f %.2f" %(global_test_acc, global_test_f1score))
        

        # Create a dictionary to store the values
        test_results = {
        "global_test_acc": global_test_acc,
        "global_test_loss": global_test_loss,
        "global_test_precision": global_test_precision,
        "global_test_recall": global_test_recall,
        "global_test_f1score": global_test_f1score
        }

                # Define the directory and filename
        directory = self.current_directory + '/results/test/' + self.fs_method 
        filename = 'dataset_%s_non_iid_%.2f_users_%d_features_%d_run_%d.pkl' % (self.dataset_name, self.noniidness, self.num_users, self.num_features, self.times)

        filepath = os.path.join(directory, filename)

        # Create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)
                # Save the dictionary to a pickle file
        with open(filepath, 'wb') as file:
            pickle.dump(test_results, file)

        return global_test_acc, global_test_f1score


    def train(self):
        loss = []
        if self.only_test:
            self.test_best_model()
        
        else:
            for glob_iter in trange(self.num_glob_iters, desc=f"features: {self.num_features} n_iid: {self.noniidness} device: {self.device}"):
                print("-------------Round number: ", glob_iter, " -------------")
                # loss_ = 0
                self.send_parameters()

            
                self.selected_users = self.select_users(glob_iter, self.num_users)
                for user in tqdm(self.selected_users, desc="processing users"):
                    user.train()  # * user.train_samples
                self.aggregate_parameters()
                # Evaluate model each interation
                self.evaluate(glob_iter)

            self.save_results()
            acc, f1s = self.test_best_model()
        
        return acc, f1s    

    def val_error_and_loss(self):
        
        accs = []
        losses = []
        precisions = []
        recalls = []
        f1s = []
        for c in self.selected_users:
            accuracy, loss, precision, recall, f1 = c.validation(self.global_model.parameters())
            accs.append(accuracy)
            losses.append(loss)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return accs, losses, precisions, recalls, f1s

    def train_error_and_loss(self):
        accs = []
        losses = []
        
        for c in self.selected_users:
            accuracy, loss = c.train_error_and_loss(self.global_model.parameters())
            accs.append(accuracy)
            losses.append(loss)
        
        return accs, losses


    def evaluate(self, t):
        val_accs, val_losses, precisions, recalls, f1s = self.val_error_and_loss()
        train_accs, train_losses  = self.train_error_and_loss()
        
        self.global_train_acc.append(statistics.mean(train_accs))
        self.global_val_acc.append(statistics.mean(val_accs))
        self.global_train_loss.append(statistics.mean(train_losses))
        self.global_val_loss.append(statistics.mean(val_losses))
        self.global_precision.append(statistics.mean(precisions))
        self.global_recall.append(statistics.mean(recalls))
        self.global_f1score.append(statistics.mean(f1s))
        


        print(f"Global Trainning Accurancy: {self.global_train_acc[t]}" )
        print(f"Global Trainning Loss: {self.global_train_loss[t]}")
        print(f"Global test accurancy: {self.global_val_acc[t]}")
        print(f"Global test_loss: {self.global_val_loss[t]}")
        print(f"Global Precision: {self.global_precision[t]}")
        print(f"Global Recall: {self.global_recall[t]}")
        print(f"Global f1score: {self.global_f1score[t]}")

        self.save_model(t)
    









    # Save loss, accurancy to h5 fiel
    def save_results(self):
        file_name = self.fs_method + "_num_users_" + str(self.num_users) + "_noniidness_" + str(self.noniidness) + "_num_features_" + str(self.num_features)
        
        print(file_name)
       
        directory_name = self.dataset_name  
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/" + directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)



        with h5py.File( self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file_name), 'w') as hf:
            hf.create_dataset('global_val_loss', data=self.global_val_loss)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_val_accuracy', data=self.global_val_acc)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('global_precieion', data=self.global_precision)
            hf.create_dataset('global_recall', data=self.global_recall)
            hf.create_dataset('global_f1score', data=self.global_f1score)

            hf.close()

            

    