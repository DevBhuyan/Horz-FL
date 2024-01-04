from global_update import server_avg
from options import args_parser
import torch
import os


def main(args):

    local_iters=args.local_iters
    global_iters=args.global_iters
    batch_size=args.batch_size
    lr = args.lr
    gpu = args.gpu
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    print("device :",device)

    csv_file = args.file_path
    dataset_name = args.dataset_name
    fs_method = args.fs_method
    num_users=args.num_users

    server = server_avg(device, dataset_name, batch_size, global_iters,
                    local_iters,num_users,lr,csv_file, fs_method)

    server.train()
    # results(args)

if __name__ == "__main__":
    args = args_parser()
    
    print("=" * 80)
    print("Summary of training process:")
    print("FS-Method: {}".format(args.fs_method))
    print("Batch size: {}".format(args.batch_size))
    print("learning rate       : {}".format(args.lr))
    print("Number of users       : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.global_iters))
    print("Number of local rounds       : {}".format(args.local_iters))
    print("Dataset name                 :{}".format(args.dataset_name))
    print("Csv file path                :{}".format(args.file_path))
    print("=" * 80)

    
    main(args)