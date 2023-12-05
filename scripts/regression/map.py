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
from tqdm import tqdm
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
        use_tqdm=args.verbose,
        return_loss=True,
        iterations=args.MAP_iterations,
        device=args.device,
    )
    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.show()

    print("MAP Loss: ", loss[-1])
    end = timer()
    torch.save(f.state_dict(), "weights/regression_weights_" + args.dataset_name)


y_mean = torch.tensor(train_dataset.targets_mean, device=args.device)
y_std = torch.tensor(train_dataset.targets_std, device=args.device)

ll_vars = [-5, -4, -3, -2, -1, 0, 1]

def get_test_step(ll_variance):

    def test_step(X, y):

        # In case targets are one-dimensional and flattened, add a final dimension.
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Cast types if needed.
        if args.dtype != X.dtype:
            X = X.to(args.dtype)
        if args.dtype != y.dtype:
            y = y.to(args.dtype)

        Fmean = f(X)  # Forward pass
        Fvar = torch.ones_like(Fmean) * np.exp(ll_variance)

        return 0, Fmean * y_std + y_mean, Fvar * y_std**2
    return test_step

best_score = np.inf
best_ll_var = None

if args.verbose:
    iters = tqdm(range(len(ll_vars)), unit = " configuration")
    iters.set_description("Finding optimal noise variance ")
else:
    iters = range(len(ll_vars))

start = timer()
for i in iters:
    f.test_step = get_test_step(ll_vars[i])

    test_metrics = score(
        f,
        val_loader,
        Regression,
        use_tqdm=False,
        device=args.device,
        dtype=args.dtype,
    )
    print(test_metrics)
    if test_metrics["NLL"] < best_score:
        best_score = test_metrics["NLL"]
        best_ll_var = ll_vars[i]
end = timer()

f.test_step = get_test_step(best_ll_var)
test_metrics = score(
    f,
    test_loader,
    Regression,
    use_tqdm=args.verbose,
    device=args.device,
    dtype=args.dtype,
)
test_metrics["val_NLL"] = best_score
test_metrics["log_variance"] = best_ll_var
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["seed"] = args.seed
test_metrics["time"] = end-start

df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()


save_str = "MAP_dataset={}_seed={}".format(
    args.dataset_name, args.seed)


df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
