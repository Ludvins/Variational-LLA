import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
import os

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map_crossentropy, score
from src.ella import ELLA_Multiclass
from utils.models import get_mlp, create_ad_hoc_mlp
from utils.metrics import SoftmaxClassification, OOD

from utils.dataset import get_dataset


args = manage_experiment_configuration()


fp = "_fixed_prior" if args.fixed_prior else ""

save_str = "ELLA_dataset={}_M={}_prior={}_seed={}{}".format(
    args.dataset_name, args.num_inducing, args.prior_std, args.seed, fp
)

if os.path.isfile("results/" + save_str + ".csv"):
    print("Experiment already exists")
    exit()


torch.manual_seed(args.seed)
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
    dropout=True,
    device=args.device,
    dtype=args.dtype,
)

# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
opt = torch.optim.SGD(f.parameters(), lr=args.MAP_lr, momentum=0.5)

# Set the number of training samples to generate

try:
    f.load_state_dict(torch.load("weights/multiclass_weights_" + args.dataset_name))
except:
    # Set the number of training samples to generate
    # Train the model
    start = timer()

    loss = fit_map_crossentropy(
        f,
        train_loader,
        opt,
        criterion=torch.nn.CrossEntropyLoss(),
        use_tqdm=True,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
        dtype=args.dtype,
    )

    print("MAP Loss: ", loss[-1])
    end = timer()
    torch.save(f.state_dict(), "weights/multiclass_weights_" + args.dataset_name)

f.eval()

ella = ELLA_Multiclass(
    create_ad_hoc_mlp(f),
    f.output_size,
    args.num_inducing,
    np.min([args.num_inducing, 20]),
    prior_std=args.prior_std,
    seed=args.seed,
    device=args.device,
    dtype=args.dtype,
)

start = timer()
ella.fit_loader_val(
            torch.tensor(train_dataset.inputs, device=args.device, dtype=args.dtype),
            torch.tensor(train_dataset.targets, device=args.device, dtype=args.dtype),
            train_loader,
            val_loader,
            val_steps = 100,
            verbose = args.verbose,
        )
end = timer()

val_metrics = score(
        ella,
        val_loader,
        SoftmaxClassification,
        use_tqdm=args.verbose,
        device=args.device,
        dtype=args.dtype,
    )

test_metrics = score(
        ella,
        test_loader,
        SoftmaxClassification,
        use_tqdm=args.verbose,
        device=args.device,
        dtype=args.dtype,
    )


test_metrics["val_NLL"] = val_metrics["NLL"]
test_metrics["prior_std"] = args.prior_std
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing
test_metrics["seed"] = args.seed
test_metrics["time"] = end-start

if args.test_ood:
    ood_dataset = dataset.get_ood_datasets()
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size)
    ood_metrics = score(
        ella, ood_loader, OOD, use_tqdm=args.verbose, device=args.device, dtype=args.dtype
    )
    test_metrics["OOD-AUC"] = ood_metrics["AUC"]
    test_metrics["OOD-AUC MC"] = ood_metrics["AUC MC"]

if args.test_corruptions:
    for corruption_value in dataset.corruption_values:
        corrupted_dataset = dataset.get_corrupted_split(corruption_value)

        loader = DataLoader(corrupted_dataset, batch_size=args.batch_size)
        corrupted_metrics = score(
            ella, loader, SoftmaxClassification, use_tqdm=args.verbose, device=args.device, dtype=args.dtype
        ).copy()
        print(corrupted_metrics)

        test_metrics = {
            **test_metrics,
            **{k+'-C'+str(corruption_value): v for k, v in corrupted_metrics.items()}
        }


df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()



print(df)



print(save_str)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
