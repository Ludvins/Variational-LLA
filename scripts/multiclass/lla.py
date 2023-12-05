from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import sys
from time import process_time as timer


sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map_crossentropy, score
from utils.models import get_mlp
from utils.metrics import SoftmaxClassification, OOD
from laplace import Laplace
import tqdm
from utils.dataset import get_dataset


args = manage_experiment_configuration()

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
        use_tqdm=args.verbose,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
        dtype=args.dtype,
    )
    print("MAP Loss: ", loss[-1])
    end = timer()
    torch.save(f.state_dict(), "weights/multiclass_weights_" + args.dataset_name)

f.eval()


# 'all', 'subnetwork' and 'last_layer'
subset = args.subset
# 'full', 'kron', 'lowrank' and 'diag'
hessian = args.hessian


la = Laplace(f, "classification", subset_of_weights=subset, hessian_structure=hessian)

inputs = torch.tensor(train_dataset.inputs, 
                      device = args.device,
                        dtype = args.dtype)


targets = torch.tensor(train_dataset.targets.squeeze(), 
                      device = args.device,
                        dtype = torch.int64)
train_dataset = TensorDataset(inputs, targets)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

start = timer()

la.fit(train_loader)

if not args.fixed_prior:
    log_prior = torch.ones(1, requires_grad=True)

    hyper_optimizer = torch.optim.Adam([log_prior], lr=1e-3)
    for i in tqdm.tqdm(range(args.iterations)):
        hyper_optimizer.zero_grad()
        neg_marglik = -la.log_marginal_likelihood(log_prior.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    #la.optimize_prior_precision(method = "CV", val_loader = val_loader, 
    #                            link_approx = "mc", verbose = args.verbose)


    prior_std = np.sqrt(1 / np.exp(log_prior.detach().numpy())).item()
else:
    prior_std = args.prior_std
end = timer()

def test_step(X, y):

        # In case targets are one-dimensional and flattened, add a final dimension.
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Cast types if needed.
        if args.dtype != X.dtype:
            X = X.to(args.dtype)
        if args.dtype != y.dtype:
            y = y.to(args.dtype)
        
        Fmean, Fvar = la._glm_predictive_distribution(X)  # Forward pass
        return 0, Fmean, Fvar


la.test_step = test_step
fp = "_fixed_prior" if args.fixed_prior else ""

save_str = "LLA_dataset={}_{}_{}_seed={}{}".format(
    args.dataset_name, subset, hessian, args.seed, fp
)


test_metrics = score(
    la,
    test_loader,
    SoftmaxClassification,
    use_tqdm=args.verbose,
    device=args.device,
    dtype=args.dtype,
)


test_metrics["prior_std"] = prior_std
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["subset"] = subset
test_metrics["hessian"] = hessian
test_metrics["seed"] = args.seed
test_metrics["time"] = end-start

print(test_metrics)

if args.test_ood:
    ood_dataset = dataset.get_ood_datasets()
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size)
    ood_metrics = score(
        la, ood_loader, OOD, use_tqdm=args.verbose, device=args.device, dtype=args.dtype
    )
    test_metrics["OOD-AUC"] = ood_metrics["AUC"]
    test_metrics["OOD-AUC MC"] = ood_metrics["AUC MC"]

if args.test_corruptions:
    for corruption_value in dataset.corruption_values:
        corrupted_dataset = dataset.get_corrupted_split(corruption_value)

        loader = DataLoader(corrupted_dataset, batch_size=args.batch_size)
        corrupted_metrics = score(
            la, loader, SoftmaxClassification, use_tqdm=args.verbose, device=args.device, dtype=args.dtype
        ).copy()
        print(corrupted_metrics)

        test_metrics = {
            **test_metrics,
            **{k+'-C'+str(corruption_value): v for k, v in corrupted_metrics.items()}
        }


df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
