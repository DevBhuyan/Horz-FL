import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fs_method", type=str, default="mrmr", choices=["no_FS",
                                                                            "multi",
                                                                            "anova",
                                                                            "rfe",
                                                                            "single",
                                                                            "mrmr"
                                                                            ])
    parser.add_argument("--dataset_name", type=str, default="vehicle")
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--batch_size", type=int, default=124)
    parser.add_argument("--lr", type=float, default=0.05, 
                        help="learning rate")
    parser.add_argument("--global_iters", type=int, default=2)
    parser.add_argument("--local_iters", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--tot_users", type=int, default=75, 
                        help="Total number of Users")
    parser.add_argument("--p", type=float, default=0.1, 
                        help="fraction of users per round")
    parser.add_argument("--non_iidness", type=float, default=1.0, 
                        help="degree of non_iidness")
    parser.add_argument("--num_features", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=26)
    parser.add_argument("--only_test", action='store_true', help="Set this flag to run only tests")
    parser.add_argument("--wandb", action='store_true')

    args = parser.parse_args()

    return args