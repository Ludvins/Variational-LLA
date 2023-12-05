from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, score
from src.ella import ELLA_Regression
from utils.models import get_mlp, create_ad_hoc_mlp
from utils.metrics import Regression
from utils.dataset import get_dataset

args = manage_experiment_configuration()

torch.manual_seed(args.seed)


if args.fixed_prior:
    save_str = "ELLA_dataset={}_M={}_ll={}_seed={}_fixed_prior".format(
        args.dataset_name, args.num_inducing, args.ll_log_var, args.seed
    )
else:
    save_str = "ELLA_dataset={}_M={}_ll={}_prior={}_seed={}".format(
        args.dataset_name, args.num_inducing, args.ll_log_var, args.prior_std, args.seed
    )
print(save_str)
import os


if os.path.isfile("results/" + save_str + ".csv"):
    print("Experiment already exists")
    exit()

dataset = get_dataset(args.dataset_name)

train_dataset, val_dataset, test_dataset = dataset.get_split(
    args.test_size, args.seed + args.split
)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


f = get_mlp(
    dataset.input_dim,
    dataset.output_dim,
    args.net_structure,
    args.activation,
    device=args.device,
    dtype=args.dtype,
)



# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()


try:
    f.load_state_dict(torch.load("weights/regression_weights_" + args.dataset_name))
except:
    # Set the number of training samples to generate
    # Train the model
    start = timer()

    loss = fit_map(
        f,
        train_loader,
        opt,
        criterion=torch.nn.MSELoss(),
        use_tqdm=True,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
    )

    print("MAP Loss: ", loss[-1])
    end = timer()
    torch.save(f.state_dict(), "weights/regression_weights_" + args.dataset_name)




ella = ELLA_Regression(
    create_ad_hoc_mlp(f),
    f.output_size,
    args.num_inducing,
    np.min([args.num_inducing, 20]),
    prior_std=args.prior_std,
    log_variance = args.ll_log_var,
    seed=args.seed,
    y_mean=train_dataset.targets_mean,
    y_std=train_dataset.targets_std,
    device=args.device,
    dtype=args.dtype,
)


start = timer()

ella.fit_loader_val(
    torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
    torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
    train_loader,
    val_loader,
    val_steps= 100,
    verbose = args.verbose,
)
end = timer()


val_metrics = score(
        ella,
        val_loader,
        Regression,
        use_tqdm=args.verbose,
        device=args.device,
        dtype=args.dtype,
    )

test_metrics = score(
        ella,
        test_loader,
        Regression,
        use_tqdm=args.verbose,
        device=args.device,
        dtype=args.dtype,
)

test_metrics["val_NLL"] = val_metrics["NLL"]
test_metrics["prior_std"] = args.prior_std
test_metrics["log_variance"] = args.ll_log_var
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing
test_metrics["seed"] = args.seed
test_metrics["time"] = end-start

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()


print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)

