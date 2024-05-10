
# %%
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# %%
import sys
import torch
from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
from properscoring import crps_gaussian

from scipy.cluster.vq import kmeans2

sys.path.append("..")

from utils.process_flags import manage_experiment_configuration
from utils.pytorch_learning import fit_map_crossentropy, fit, forward, score
from utils.models import get_resnet
from utils.dataset import get_dataset
from utils.metrics import SoftmaxClassification, OOD
from src.utils import smooth
import matplotlib.pyplot as plt

from utils.dataset import get_dataset
import argparse

# %%

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--seed', type=int)
parser.add_argument('--resnet', type=str)
parser.add_argument('--prior', type=float)
args = parser.parse_args()


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
torch.backends.cudnn.benchmark = True

# %%
dataset = get_dataset("CIFAR10")

# %%
torch.manual_seed(args.seed)

train_dataset, val_dataset, test_dataset = dataset.get_split(
    0.1, args.seed
)

# %%
rng = np.random.RandomState(args.seed)
subset = rng.choice(len(train_dataset), 500, replace=False)

subset_inputs = train_dataset.inputs[subset]
subset_targets = train_dataset.targets[subset]

subset_inputs = torch.tensor(subset_inputs, device=device, dtype = torch.float32)
subset_targets = torch.tensor(subset_targets, device=device, dtype=torch.long)

# %%
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# %%
import torch

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import BatchGrad
from backpack.context import CTX
from laplace.curvature import CurvatureInterface


class BackPackInterface(CurvatureInterface):
    """Interface for Backpack backend."""

    def __init__(self, model, output_dim):
        super().__init__(model, "regression", False, None)
        extend(self._model)
        self.output_size = output_dim

    def jacobians(self, x, enable_back_prop=False):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per output dimension.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, input_shape)
            input data on compatible device with model.
        enable_back_prop : boolean
            If True, computational graph is retained and backpropagation can be used
            on elements used in this function.

        Returns
        -------
        Js : torch.Tensor of shape (batch, parameters, outputs)
            Jacobians ``
        """
        # Enable grads in this section of code
        with torch.set_grad_enabled(True):
            # Extend model using BackPack converter
            model = extend(self.model, use_converter=True)
            # Set model in evaluation mode to ignore Dropout, BatchNorm..
            model.eval()

            # Initialice array to concatenate
            to_stack = []

            # Loop over output dimension
            for i in range(self.output_size):
                # Reset gradients
                model.zero_grad()
                # Compute output
                out = model(x)  

                # Use Backpack Gradbatch to retain independent gradients for each input.
                with backpack(BatchGrad()):
                    # Compute backward pass on the corresponding output (if more than one)
                    if self.output_size > 1:
                        out[:, i].sum().backward(
                            create_graph=enable_back_prop, retain_graph=enable_back_prop
                        )
                    else:
                        out.sum().backward(
                            create_graph=enable_back_prop, retain_graph=enable_back_prop
                        )
                    # Auxiliar array
                    to_cat = []
                    # Loop over model parameters, retrieve their gradient and delete it
                    for param in model.parameters():
                        to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                        delattr(param, "grad_batch")
                    # Stack all gradientsÇ
                    Jk = torch.cat(to_cat, dim=1)
                # Append result
                if i == 0:
                    to_stack = Jk.unsqueeze(0)
                else:
                    to_stack = torch.cat([to_stack, Jk.unsqueeze(0)], 0)
                #to_stack.append(Jk)

        # Clean model gradients
        model.zero_grad()
        # Erase gradients form input
        x.grad = None
        # Clean BackPak hooks
        CTX.remove_hooks()
        # Clean extended model
        _cleanup(model)

        # Return Jacobians
        if self.output_size > 1:
            return to_stack.transpose(0, 1)
        else:
            return Jk.unsqueeze(-1).transpose(1, 2)
def _cleanup(module):
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)

    

# %%
from tqdm import tqdm
from utils.metrics import _ECELoss

# %%
f = get_resnet(args.resnet, 10).to(device).to(torch.float32)
backend = BackPackInterface(f, train_dataset.output_dim)
f.eval()

with torch.no_grad():
    
    # Compute Jacobian
    x = subset_inputs.to(device)
    y = subset_targets.to(device)
    # Shape (num_data, num_classes)
    predictions = f(x)
    probs = torch.nn.functional.softmax(predictions, dim=1)
        
    # Shape (num_data, num_classes, num_classes)
    likelihood_hessian = torch.diag_embed(probs) - torch.einsum('bi,bj->bij', probs, probs)
    # Shape (num_data, num_classes, num_classes)    
    likelihood_hessian_inv = torch.inverse(likelihood_hessian)
    
    # Shape (num_classes, num_classes, num_data, num_data)
    likelihood_hessian_inv = torch.diag_embed(likelihood_hessian_inv.permute(1, 2, 0))
    
    # Shape (num_data, num_data, num_classes, num_classes)
    likelihood_hessian_inv = likelihood_hessian_inv.permute(2, 3, 0, 1)
    
    # Shape (num_data, num_params, num_classes)
    Jx = backend.jacobians(x, enable_back_prop=False)
    
    # Shape (num_data, num_classes, num_data, num_classes)
    Kxx = args.prior * torch.einsum('nap,map -> nma', Jx, Jx)
    
    # Shape (num_data, num_data, num_classes, num_classes)
    Kxx = torch.diag_embed(Kxx)
    
    # Shape (num_data, num_data, num_classes, num_classes)
    Q = likelihood_hessian_inv + Kxx
    Q = Q.permute(0,2,1,3)
    N, O = Q.shape[0], Q.shape[1]
    # Flatten
    Q = Q.flatten(0,1).flatten(1,2)
    # Shape (num_data*num_classes, num_data*num_classes)
    Q_inv = torch.inverse(Q)
    
    Q_inv = Q_inv.view(N, O, N, O)
    
    NLL = 0
    ACC = 0
    ECE = _ECELoss()
    
    NLL_val = 0
    ACC_val = 0
    ECE_val = _ECELoss()
    
    iterator = iter(test_loader)
    
    for it in tqdm(range(len(test_loader))):
        
        test_x, test_y = next(iterator)
        
        test_x = test_x.to(device).to(torch.float32)
        test_y = test_y.to(device)
        
        Jt = backend.jacobians(test_x, enable_back_prop=False)
        
        Kxz = args.prior * torch.einsum('nap,mbp->nmab', Jx, Jt)
        
        Kzz = args.prior * torch.einsum('nap,nbp->nab', Jt, Jt)
        
        K2 =  torch.einsum('nmab,nalc,lmcd->mad', Kxz, Q_inv, Kxz) 
                
        var = Kzz-K2
        mean = f(test_x)
        
        # Create torch generator
        generator = torch.Generator(device)
        
        z = torch.randn(2048, mean.shape[0], var.shape[-1], generator = generator, device=device)        
        chol = torch.linalg.cholesky(var)
        
        samples = mean + torch.einsum("sna, nab -> snb", z, chol)
        probs = torch.nn.functional.softmax(samples, dim=-1)
        probs = probs.mean(dim=0)
        
        ll = torch.nn.functional.cross_entropy(probs.log(), 
                                                test_y.to(torch.long).squeeze(-1), 
                                                reduction = "sum")
        acc = (probs.argmax(dim=-1) == test_y.to(torch.long).squeeze(-1)).sum()
        
        ECE.update(test_y.to(torch.long).squeeze(-1), probs)
        
        NLL += ll
        
        ACC += acc
        

    iterator = iter(val_loader)
    
    for it in tqdm(range(len(val_loader))):
        
        test_x, test_y = next(iterator)
        
        test_x = test_x.to(device).to(torch.float32)
        test_y = test_y.to(device)
        
        Jt = backend.jacobians(test_x, enable_back_prop=False)
        
        Kxz = args.prior * torch.einsum('nap,mbp->nmab', Jx, Jt)
        
        Kzz = args.prior * torch.einsum('nap,nbp->nab', Jt, Jt)
        
        K2 =  torch.einsum('nmab,nalc,lmcd->mad', Kxz, Q_inv, Kxz) 
                
        var = Kzz-K2
        mean = f(test_x)
        
        # Create torch generator
        generator = torch.Generator(device)
        
        z = torch.randn(2048, mean.shape[0], var.shape[-1], generator = generator, device=device)        
        chol = torch.linalg.cholesky(var)
        
        samples = mean + torch.einsum("sna, nab -> snb", z, chol)
        probs = torch.nn.functional.softmax(samples, dim=-1)
        probs = probs.mean(dim=0)
        
        ll = torch.nn.functional.cross_entropy(probs.log(), 
                                                test_y.to(torch.long).squeeze(-1), 
                                                reduction = "sum")
        acc = (probs.argmax(dim=-1) == test_y.to(torch.long).squeeze(-1)).sum()
        
        ECE_val.update(test_y.to(torch.long).squeeze(-1), probs)
        
        NLL_val += ll
        
        ACC_val += acc
        
    
    NLL = NLL.detach().cpu().numpy().item()/len(test_dataset)
    ACC  = ACC.detach().cpu().numpy().item()/len(test_dataset)
    ECE = ECE.compute().detach().cpu().numpy().item()
    NLL_val = NLL_val.detach().cpu().numpy().item()/len(val_dataset)
    ACC_val  = ACC_val.detach().cpu().numpy().item()/len(val_dataset)
    ECE_val = ECE_val.compute().detach().cpu().numpy().item()
    
    print("NLL: ", NLL)
    print("ACC: ", ACC)
    print("ECE: ", ECE)
    
    s = np.array([ACC_val, NLL_val, ECE_val, ACC, NLL, ECE])
    import os
    
    np.savetxt(os.path.join("results/results_{}_{}_{}.txt".format(args.resnet, args.seed, args.prior)), s)

# %%



