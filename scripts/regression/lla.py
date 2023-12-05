import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map, score
from utils.models import get_mlp
from utils.metrics import Regression
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
    device=args.device,
    dtype=args.dtype,
)


# Define optimizer and compile model
opt = torch.optim.Adam(f.parameters(), lr=args.MAP_lr, weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()

# Set the number of training samples to generate

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


# 'all', 'subnetwork' and 'last_layer'
subset = args.subset
# 'full', 'kron', 'lowrank' and 'diag'
hessian = args.hessian
X = test_dataset.inputs
la = Laplace(f, "regression", subset_of_weights=args.subset, hessian_structure=args.hessian)

start = timer()

la.fit(train_loader)

log_sigma = torch.ones(1, requires_grad=True)
if not args.fixed_prior:
    log_prior = torch.ones(1, requires_grad=True)
else:
    log_prior = torch.ones(1, requires_grad = False) * np.log((1/(args.prior_std**2)))

hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-3)

if args.verbose:
    it = tqdm.tqdm(range(args.iterations))
else:
    it = range(args.iterations)


for i in it:
    hyper_optimizer.zero_grad()
    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

end = timer()

prior_std = np.sqrt(1 / np.exp(log_prior.detach().numpy())).item()
log_variance = 2*log_sigma.detach().numpy().item()



y_mean = torch.tensor(train_dataset.targets_mean, device=args.device)
y_std = torch.tensor(train_dataset.targets_std, device=args.device)



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
        Fvar = Fvar + np.exp(log_variance)
        return 0, Fmean * y_std + y_mean, Fvar * y_std**2


la.test_step = test_step
fp = "_fixed_prior" if args.fixed_prior else ""

save_str = "LLA_dataset={}_{}_{}_seed={}{}".format(
    args.dataset_name, args.subset, args.hessian, args.seed, fp
)


test_metrics = score(
    la,
    test_loader,
    Regression,
    use_tqdm=args.verbose,
    device=args.device,
    dtype=args.dtype,
)

test_metrics["prior_std"] = prior_std
test_metrics["log_variance"] = log_variance
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["subset"] = args.subset
test_metrics["hessian"] = args.hessian
test_metrics["seed"] = args.seed
test_metrics["time"] = end-start

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

print(df)

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
