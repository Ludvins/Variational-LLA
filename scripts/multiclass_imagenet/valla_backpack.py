from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from properscoring import crps_gaussian

from scipy.cluster.vq import kmeans2

sys.path.append(".")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map_crossentropy, fit, forward, score
from scripts.filename import create_file_name
from src.valla import VaLLAMultiClassBackend
from utils.models import get_resnet
from utils.dataset import imagenet_loaders
from utils.metrics import SoftmaxClassification, OOD
from src.utils import smooth
import matplotlib.pyplot as plt
args = manage_experiment_configuration()

torch.manual_seed(args.seed)
train_loader, test_loader, train_dataset = imagenet_loaders(args, valid_size=0.0)

import torchvision.models as models

f = models.__dict__[args.resnet](pretrained=True).to(args.device).to(args.dtype)
f.eval()

rng = np.random.default_rng(args.seed)

indexes = rng.choice(np.arange(len(train_dataset)), args.num_inducing, replace = False)
Z = []
classes = []
for index in indexes:
    img, label = train_dataset[index]
    Z.append(img)
    classes.append(label)
Z = np.stack(Z, 0)
classes = np.stack(classes, 0)

from src.backpack_interface import BackPackInterface
valla = VaLLAMultiClassBackend(
    f,
    Z,
    backend = BackPackInterface(f, 1000),
    prior_std=args.prior_std,
    num_data=len(train_dataset),
    output_dim=1000,
    track_inducing_locations=False,
    inducing_classes=classes,
    y_mean=0,
    y_std=1,
    alpha = args.bb_alpha,
    device=args.device,
    dtype=args.dtype,
    #seed = args.seed
)


opt = torch.optim.Adam(valla.parameters(recurse=False), lr=args.lr)

start = timer()
loss, val_loss = fit(
    valla,
    train_loader,
    opt,
    val_metrics=SoftmaxClassification,
    val_steps=100,
    val_generator = test_loader,
    use_tqdm=args.verbose,
    return_loss=True,
    iterations=args.iterations,
    device=args.device,
    dtype = args.dtype
)
print(val_loss)
end = timer()


save_str = "VaLLA_{}_dataset=ImageNet_M={}_seed={}".format(
    args.resnet, args.num_inducing, args.seed
)


plt.savefig("plots/" + save_str + ".pdf")

test_metrics = score(
    valla,
    test_loader,
    SoftmaxClassification,
    use_tqdm=args.verbose,
    device=args.device,
    dtype=args.dtype,
)

test_metrics["prior_std"] = np.exp(valla.log_prior_std.detach().cpu().numpy())
test_metrics["iterations"] = args.iterations
test_metrics["weight_decay"] = args.weight_decay
test_metrics["dataset"] = args.dataset_name
test_metrics["MAP_iterations"] = args.MAP_iterations
test_metrics["M"] = args.num_inducing
test_metrics["seed"] = args.seed
test_metrics["alpha"] = args.bb_alpha
test_metrics["timer"] = end-start

if args.test_ood:
    ood_dataset = args.dataset.get_ood_datasets()
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size)
    ood_metrics = score(
        valla, ood_loader, OOD, use_tqdm=args.verbose, device=args.device, dtype=args.dtype
    )
    test_metrics["OOD-AUC"] = ood_metrics["AUC"]
    test_metrics["OOD-AUC MC"] = ood_metrics["AUC MC"]

if args.test_corruptions:
    for corruption_type in range(args.dataset.corruption_types):
        for corruption_value in range(args.dataset.corruption_values):
            corrupted_dataset = args.dataset.get_corrupted_split(corruption_type, corruption_value)

            loader = DataLoader(corrupted_dataset, batch_size=args.batch_size)
            corrupted_metrics = score(
                valla, loader, SoftmaxClassification, use_tqdm=args.verbose,
                device=args.device, dtype=args.dtype
            ).copy()

            test_metrics = {
                **test_metrics,
                **{k+'-C-'+str(corruption_type)+"-"+str(corruption_value): v 
                   for k, v in corrupted_metrics.items()}
            }



df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()

df.to_csv(
    path_or_buf="results/" + save_str + ".csv",
    encoding="utf-8",
)
