import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fs_method", type=str, default="no_FS", choices=["no_FS",
                                                                            "fed_mofs",
                                                                            "anova",
                                                                            "rfe",
                                                                            "fed_fis"
                                                                            ])
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05, 
                        help="learning rate")
    parser.add_argument("--global_iters", type=int, default=200)
    parser.add_argument("--local_iters", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--num_users", type=int, default=10, 
                        help="Number of Users per round")
    args = parser.parse_args()

    return args