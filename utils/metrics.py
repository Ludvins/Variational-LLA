#!/usr/bin/env python3
import numpy as np
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from sklearn.metrics import roc_auc_score
from properscoring import crps_gaussian
from scipy.stats import norm
from src.utils import psd_safe_cholesky

class Metrics:
    def __init__(self, num_data=-1, device=None):
        """Defines a class that encapsulates all considered metrics.

        Arguments
        ---------
        num_data : int
                   Number of data samples in the dataset used. This is used
                   to scale the metrics from each batch by the proportion of
                   data they contain.
                   This scale is never greater than 1.
        device : torch device
                 Device in which to make the computations.
        """

        self.num_data = num_data
        self.device = device
        self.reset()

    def reset(self):
        """Ressets all the metrics to zero."""
        self.loss = torch.tensor(0.0, device=self.device)
        self.nll = torch.tensor(0.0, device=self.device)

    def update(self, y, loss, mean_pred, std_pred, likelihood, light=True):
        raise NotImplementedError

class RegressionNLL(Metrics):
    def __init__(self, num_data, device, dtype):
        super().__init__(num_data, device)
        self.dtype = dtype

    def update(self, y, loss, Fmean, Fvar):
        # Conmpute the scale value using the batch_size
        batch_size = y.shape[0]
        if self.num_data == -1:
            scale = 1
        else:
            scale = batch_size / self.num_data

        self.nll += scale * self.compute_nll(y, Fmean, Fvar)

    def compute_nll(self, y, Fmean, Fvar):
        var = Fvar.squeeze()
        ll = -0.5 * (np.log(2 * np.pi) + torch.log(var) + (Fmean.squeeze() - y.squeeze())**2 / var)
        return torch.mean(-ll)

    def get_dict(self):
        return self.nll
        

class Regression(Metrics):
    def __init__(self, num_data, device, dtype):
        super().__init__(num_data, device)
        self.dtype = dtype

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.mse = torch.tensor(0.0, device=self.device)
        self.crps = torch.tensor(0.0, device=self.device)

        self.q_10 = torch.tensor(0.0, device=self.device)
        self.q_20 = torch.tensor(0.0, device=self.device)
        self.q_30 = torch.tensor(0.0, device=self.device)
        self.q_40 = torch.tensor(0.0, device=self.device)
        self.q_50 = torch.tensor(0.0, device=self.device)
        self.q_60 = torch.tensor(0.0, device=self.device)
        self.q_70 = torch.tensor(0.0, device=self.device)
        self.q_80 = torch.tensor(0.0, device=self.device)
        self.q_90 = torch.tensor(0.0, device=self.device)


    def update(self, y, loss, Fmean, Fvar):
        # Conmpute the scale value using the batch_size
        batch_size = y.shape[0]
        if self.num_data == -1:
            scale = 1
        else:
            scale = batch_size / self.num_data
        # Update light metrics
        self.loss += scale * loss
        # The RMSE is computed using the mean prediction of the Gaussian
        #  Mixture, that is, the mean of the mean predictions.
        self.mse += scale * self.compute_mse(y, Fmean)
        self.nll += scale * self.compute_nll(y, Fmean, Fvar)
        
        Fstd = torch.sqrt(Fvar)
        self.crps += scale * self.compute_crps(y, Fmean, Fstd)

        self.q_10 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.10)
        self.q_20 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.20)
        self.q_30 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.30)
        self.q_40 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.40)
        self.q_50 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.50)
        self.q_60 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.60)
        self.q_70 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.70)
        self.q_80 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.80)
        self.q_90 += scale * self.compute_quantile_error(y, Fmean, Fstd, 0.90)

    def compute_quantile_error(self, y, Fmean, Fstd, alpha):
        deviation = norm.ppf(0.5 + alpha/2)
        u = (Fmean + deviation*Fstd)
        l = (Fmean - deviation*Fstd)
        inside = ((y < u) * (y > l)).to(torch.float32)

        #deviation = norm.ppf(alpha)
        #l = Fmean + deviation * Fstd
        #inside = (y < l).to(torch.float32)
        return torch.mean(inside)


    def compute_mse(self, y, prediction):
        """Computes the root mean squared error for the given predictions."""
        return torch.nn.functional.mse_loss(prediction, y)
    
    def compute_nll(self, y, Fmean, Fvar):
        var = Fvar.squeeze()
        ll = -0.5 * (np.log(2 * np.pi) + torch.log(var) + (Fmean.squeeze() - y.squeeze())**2 / var)
        return torch.mean(-ll)

    def compute_crps(self, y, mean_pred, std_pred):
        crps= crps_gaussian(y.detach().cpu(), mean_pred.detach().cpu(), std_pred.detach().cpu())
        return np.mean(np.sum(crps, -1))

    def get_dict(self):
        return {
            "LOSS": float(self.loss.detach().cpu().numpy()),
            "RMSE": np.sqrt(self.mse.detach().cpu().numpy()),
            "NLL": float(self.nll.detach().cpu().numpy()),
            "Q-10": float(self.q_10.detach().cpu().numpy()),
            "Q-20": float(self.q_20.detach().cpu().numpy()),
            "Q-30": float(self.q_30.detach().cpu().numpy()),
            "Q-40": float(self.q_40.detach().cpu().numpy()),
            "Q-50": float(self.q_50.detach().cpu().numpy()),
            "Q-60": float(self.q_60.detach().cpu().numpy()),
            "Q-70": float(self.q_70.detach().cpu().numpy()),
            "Q-80": float(self.q_80.detach().cpu().numpy()),
            "Q-90": float(self.q_90.detach().cpu().numpy()),
            "CRPS": float(self.crps),
        }


class _ECELoss(torch.nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        
        self.confidences = []
        self.predictions = []
        self.labels = []
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        bin_boundaries_plot = torch.linspace(0, 1, 11)
        self.bin_lowers_plot = bin_boundaries_plot[:-1]
        self.bin_uppers_plot = bin_boundaries_plot[1:]
        
    def reset(self):
        self.confidences = []
        self.predictions = []
        self.labels = []
    
    def update(self, labels, F):
        probs = F.softmax(-1)
        conf, pred = torch.max(probs, -1)
        self.confidences.append(conf)
        self.predictions.append(pred)
        self.labels.append(labels.squeeze(-1))

    def compute(self):
        
        self.predictions = torch.cat(self.predictions, -1)
        self.labels = torch.cat(self.labels, -1)
        self.confidences = torch.cat(self.confidences, -1)
        
        
        accuracies = self.predictions.eq(self.labels)
        ece = torch.zeros(1, device=self.confidences.device)
        
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = self.confidences.gt(bin_lower.item()) * self.confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = self.confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class SoftmaxClassificationNLL(Metrics):
    def __init__(self, num_data=-1, device=None, dtype = None):
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(2147483647)
        super().__init__(num_data, device)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.generator.manual_seed(2147483647)
        self.nll = torch.tensor(0.0, device=self.device)
        
    def update(self, y, loss, Fmean, Fvar):
        """Updates all the metrics given the results in the parameters.

        Arguments
        ---------

        y : torch tensor of shape (batch_size, output_dim)
            Contains the true targets of the data.
        loss : torch tensor of shape ()
               Contains the loss value for the given batch.
        likelihood : instance of Likelihood
                     Usable to compute the log likelihood metric.
        """
        # Conmpute the scale value using the batch_size
        batch_size = y.shape[0]
        if self.num_data == -1:
            scale = 1
        else:
            scale = batch_size / self.num_data
               
        chol = psd_safe_cholesky(Fvar)
        z = torch.randn(2048, Fmean.shape[0], Fvar.shape[-1], generator = self.generator, device=self.device, dtype = self.dtype)
        samples = Fmean + torch.einsum("sna, nab -> snb", z, chol)
        
        probs = samples.softmax(-1)
        mean = probs.mean(0)
        self.nll += scale * self.compute_nll(y, mean.log())
    
    def compute_nll(self, y, F):
        nll = torch.nn.functional.cross_entropy(F, y.to(torch.long).squeeze(-1), reduction = "none")
        return nll.mean()

    def get_dict(self):
        return self.nll

class SoftmaxClassification(Metrics):
    def __init__(self, num_data=-1, device=None, dtype = None):
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(2147483647)
        self.dtype = dtype
        super().__init__(num_data, device)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.nll_mc = torch.tensor(0.0, device=self.device)
        self.acc_mc = torch.tensor(0.0, device=self.device)
        self.brier_mc = torch.tensor(0.0, device=self.device)
        self.ece_mc = _ECELoss()
        self.nll = torch.tensor(0.0, device=self.device)
        self.acc = torch.tensor(0.0, device=self.device)
        self.brier = torch.tensor(0.0, device=self.device)
        self.ece = _ECELoss()

        self.generator.manual_seed(2147483647)
        
        
    def update(self, y, loss, Fmean, Fvar):
        """Updates all the metrics given the results in the parameters.

        Arguments
        ---------

        y : torch tensor of shape (batch_size, output_dim)
            Contains the true targets of the data.
        loss : torch tensor of shape ()
               Contains the loss value for the given batch.
        likelihood : instance of Likelihood
                     Usable to compute the log likelihood metric.
        """
        # Conmpute the scale value using the batch_size
        batch_size = y.shape[0]
        if self.num_data == -1:
            scale = 1
        else:
            scale = batch_size / self.num_data
        # Update light metrics
         
        self.loss += scale * loss
        
        chol = psd_safe_cholesky(Fvar)
        z = torch.randn(2048, Fmean.shape[0], Fvar.shape[-1], generator = self.generator, device = self.device,
                            dtype = self.dtype)
        samples = Fmean + torch.einsum("sna, nab -> snb", z, chol)
        
        probs = samples.softmax(-1)
        mean = probs.mean(0)
        self.acc_mc += scale * self.compute_acc(y, mean)
        self.nll_mc += scale * self.compute_nll(y, mean.log())
        self.brier_mc += scale * self.compute_brier(y, mean.log())
        self.ece_mc.update(y, mean.log())
        
        
        scaled_logits = Fmean/torch.sqrt( 1 + torch.pi/8 * torch.diagonal(Fvar, dim1 = 1, dim2 = 2))

        self.acc += scale * self.compute_acc(y, scaled_logits)
        self.nll += scale * self.compute_nll(y, scaled_logits)
        self.brier += scale * self.compute_brier(y,scaled_logits)

        self.ece.update(y, scaled_logits)
        
    def compute_brier(self, y, F):
        probs = F.softmax(-1)

        oh_on = torch.nn.functional.one_hot(y.squeeze(), num_classes = probs.shape[-1])
        dist = (probs - oh_on)**2
        dist = torch.sum(dist, -1)
        return torch.mean(dist)
        
    def compute_acc(self, y, F):
        # F shape (..., num_classes)
        
        # Shape (...)
        pred = torch.argmax(F, -1)
        
        return (pred == y.flatten()).float().mean()
    
    def compute_nll(self, y, F):
        nll = torch.nn.functional.cross_entropy(F, y.to(torch.long).squeeze(-1), reduction = "none")
        return nll.mean()


    def get_dict(self):
        return {
            "LOSS": float(self.loss.detach().cpu().numpy()),
            "NLL": float(self.nll.detach().cpu().numpy()),
            "ACC": float(self.acc.detach().cpu().numpy()),
            "ECE": float(self.ece.compute().detach().cpu().numpy()),
            "BRIER": float(self.brier.detach().cpu().numpy()),
            "NLL MC": float(self.nll_mc.detach().cpu().numpy()),
            "ACC MC": float(self.acc_mc.detach().cpu().numpy()),
            "ECE MC": float(self.ece_mc.compute().detach().cpu().numpy()),
            "BRIER MC": float(self.brier_mc.detach().cpu().numpy()),
        }



class OOD(Metrics):
    def __init__(self, num_data=-1, device=None, dtype = None):
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(2147483647)
        self.dtype = dtype
        super().__init__(num_data, device)

    def reset(self):
        """Ressets all the metrics to zero."""
        super().reset()
        self.labels = []
        self.preds_mc = []
        self.preds = []

        self.generator.manual_seed(2147483647)
        
        
    def update(self, y, loss, Fmean, Fvar):
        """Updates all the metrics given the results in the parameters.

        Arguments
        ---------

        y : torch tensor of shape (batch_size, output_dim)
            Contains the true targets of the data.
        loss : torch tensor of shape ()
               Contains the loss value for the given batch.
        likelihood : instance of Likelihood
                     Usable to compute the log likelihood metric.
        """
        with torch.no_grad():            
            chol = psd_safe_cholesky(Fvar)
            z = torch.randn(2048, Fmean.shape[0], Fvar.shape[-1], generator = self.generator,
                            device = self.device, dtype = self.dtype)
            samples = Fmean + torch.einsum("sna, nab -> snb", z, chol)
            
            # Compute probabilities
            probs = samples.softmax(-1)
            # Average on sample dimension
            probs = probs.mean(0)
            # Compute Entropy
            H = - torch.sum(probs * probs.log(), -1)       
            # Store Monte-Carlo entropy
            self.preds_mc.append(H)
            
            # Compute deterministic scaled logits
            scaled_logits = Fmean/torch.sqrt( 1 + torch.pi/8 * torch.diagonal(Fvar, dim1 = 1, dim2 = 2))
            # Compute probabilities
            probs = scaled_logits.softmax(-1)

            # Compute Deterministic Entropy
            H = - torch.sum(probs * probs.log(), -1)       
            # Store
            self.preds.append(H)
            
            # Store labels
            self.labels.append(y)
        
    def get_dict(self):
        self.labels = torch.cat(self.labels)
        self.preds = torch.cat(self.preds)
        self.preds_mc = torch.cat(self.preds_mc)

        auc = roc_auc_score(self.labels.squeeze(-1).detach().cpu().numpy(), 
                            self.preds.detach().cpu().numpy())
        auc_mc = roc_auc_score(self.labels.squeeze(-1).detach().cpu().numpy(), 
                               self.preds_mc.detach().cpu().numpy())

        return {
            "AUC MC": auc_mc,
            "AUC": auc,
        }
