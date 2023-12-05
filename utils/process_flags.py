import argparse

import numpy as np
import torch

from .dataset import get_dataset


def manage_experiment_configuration(args=None):

    if args is None:
        # Get parser arguments
        parser = get_parser()
        args = parser.parse_args()

    FLAGS = vars(args)

    FLAGS["activation_str"] = args.activation

    if args.activation == "tanh":
        FLAGS["activation"] = torch.nn.Tanh
    elif args.activation == "relu":
        FLAGS["activation"] = torch.nn.ReLU
    else:
        raise ValueError("Invalid BNN activation type.")

    if args.device == "gpu":
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("Enabling GPU usage")

        args.device = torch.device("cuda" if use_cuda else "cpu")
        print("Device: ", args.device)
        torch.backends.cudnn.benchmark = True

    if args.dtype == "float64":
        FLAGS["dtype"] = torch.float64
    if args.dtype == "float32":
        FLAGS["dtype"] = torch.float32   

    return args


def get_parser():
    """
    Defines and returns a parser for DeepVIP experiments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
    )

    parser.add_argument(
        "--hessian",
        type=str,
    )

    parser.add_argument(
        "--subset",
        type=str,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Training iterations",
    )

    parser.add_argument(
        "--MAP_iterations",
        type=int,
        help="Training iterations",
    )    

    parser.add_argument(
        "--bb_alpha",
        type=float,
        default=0,
        help="Value of alpha in alpha energy divergence.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size",
    )
    parser.add_argument(
        "--num_inducing",
        type=int,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Training learning rate",
    )


    parser.add_argument(
        "--MAP_lr",
        type=float,
        default=0.001,
        help="MAP Training learning rate",
    )

    parser.add_argument(
        "--net_structure",
        type=int,
        default=[200, 200],
        nargs="+",
        help="Specifies the width of the inner layers of the underlying model.",
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        help=(
            "Activation function to use in the Bayesian NN. Options:"
            "tanh, relu, sigmoid, cos"
        ),
    )

    parser.add_argument(
        "--resnet",
        type=str,
        default="resnet20",
        help=(
            "Activation function to use in the Bayesian NN. Options:"
            "tanh, relu, sigmoid, cos"
        ),
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Test proportion for datasets with non defined train test splits.",
    )

    parser.add_argument(
        "--prior_std",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--ll_log_var",
        type=float,
        default=-5,
    )


    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--sub_classes",
        type=int,
        default=-1,
    )


    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--test_ood",
        dest="test_ood",
        action="store_true",
    )
    parser.set_defaults(test_ood=False)

    parser.add_argument(
        "--test_corruptions",
        dest="test_corruptions",
        action="store_true",
    )
    parser.set_defaults(test_corruptions=False)


    parser.add_argument(
        "--fixed_prior",
        dest="fixed_prior",
        action="store_true",
    )
    parser.set_defaults(fixed_prior=False)

    parser.add_argument(
        "--fix_inducing",
        dest="fix_inducing",
        action="store_true",
    )
    parser.set_defaults(fix_inducing=False)

    parser.add_argument("--seed", type=int, default=2147483647, help="Random seed.")

    parser.add_argument("--dtype", type=str, default=torch.float64, help="Data type")
    parser.add_argument("--split", default=None, type=int, help="Data split to use.")

    return parser
