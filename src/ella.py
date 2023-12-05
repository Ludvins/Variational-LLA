#!/usr/bin/env python3

import torch
import numpy as np
from utils.pytorch_learning import score
from utils.metrics import SoftmaxClassificationNLL, RegressionNLL
from src.backpack_interface import BackPackInterface


class ELLA_Base(torch.nn.Module):
    """Contains an accElerated Linearized Laplace Aproximation model.

    Parameters
    ----------
        net : callable
              Deterministic deep learning moden in which to a apply LLA. Must
              have methods "jacobians_on_outputs" to compute its jacobians.
        output_size : int
                      Number of output dimenions of the deep learning model.
        n_samples : int
                    Amount of MC samples to take from the training dataset
                    to approximate the kernel matrix.
        n_eigh : int
                 Number of eigenvalues to consider for the features
                 approximation.
        prior_std : float
                    Parameters' prior standard deviation of their isotropic
                    Gaussian distribution.
        seed : int
               Random seed to ensure reproducilibity of the code.
        y_mean : float
                 Mean of the target values, used to re-escale outputs.
        y_std : float
                Standard deviation of the target_values, used to re-escale oututs.
        device : torch.device
                 Device in which to perform inference.
        dtype : torch.dtype
                Precision used for the model.
    """

    def __init__(
        self,
        net,
        output_size,
        n_samples,
        n_eigh,
        prior_std,
        seed,
        y_mean,
        y_std,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()

        # Initialize model and output size
        self.output_size = output_size
        self.net = net

        # Store ELLA constants
        self.M = n_samples
        self.K = n_eigh

        # Store target constants
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)

        # Seed
        self.seed = seed

        # Prior
        self.prior_std = torch.tensor(prior_std, device=device, dtype=dtype)

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype

    def test_step(self, X, y):
        """
        Defines the test step for the ELLA model.
        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input.
        Returns
        -------
        loss : float
               The nelbo of the model at the current state for the given inputs.
        Fmean : torch tensor of size (batch_size, output_dim)
                Predictive mean of the model on the given batch
        Fvar : torch tensor of size (batch_size, output_dim, output_dim)
               Contains the covariance matrix of the model for each element on
               the batch.
        """

        # In case targets are one-dimensional and flattened, add a final dimension.
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Cast types if needed.
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        Fmean, Fvar = self(X)  # Forward pass

        return 0, Fmean, Fvar

    def predict_f(self, X):
        """
        Defines the predictive computation of ELLA.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        Returns
        -------
        mean : torch tensor of size (batch_size, output_dim)
               Predictive mean of the model on the given batch
        K : torch tensor of size (batch_size, output_dim, output_dim)
            Contains the covariance matrix of the model for each element on
            the batch.
        """

        # No grad is required for ELLA computations
        with torch.no_grad():
            # Compute G inverse adding the prior.
            prior = torch.eye(self.K, device=self.device, dtype=self.dtype) / (
                self.prior_std**2
            )
            G_inv = torch.inverse(self.G + prior)

            # Get Jacobians with shape (batch_size, output_size, n_parameters)
            J = self.net.get_jacobian(X)

            # Compute low-dimensional features as J @ v
            # self.v has shape (n_parameters, self.n_eigh)
            # low-dimensional features are (batch_size, output_dim, self.n_eigh)
            phi = torch.einsum("mds, sk -> mdk", J, self.v)

            # Covariance matrix is computed as phi @ G^{-1} @ phi.T
            # n is batch_size
            # a and b is output dimension
            # k and g are self.n_eigh dimension
            K = torch.einsum("nak, kg, nbg -> nab", phi, G_inv, phi)

            # Compute mean using network
            mean = self.net(X)

        return mean, K

    def likelihood_hessian(self, data, target):
        """Computes the hessian of the likelihood function wrt the model output."""
        raise NotImplementedError

    def fit_loader_val(
        self, X_train, y_train, train_loader, val_loader, val_steps=1, verbose=True
    ):
        """Fits ELLA model to the given training dataset. Performs early-stopping using
        a validation dataset.

        Parametes
        ---------
            X_train : torch.tensor of shape (n_train, input_shape)
                      Contains the input dataset.
            y_train : torch.tensor of shape (n_train, )
                      Contains the input labels.
            train_loader : torch.datasets.loader
                           Dataloader instance of the same training dataset
                           provided on (X_train, y_train).
            val_loader : torch.datasets.loader
                        Dataloader instance of a validation dataset.
            val_steps : int
                        Number of iterations between validations.
            verbose : boolean
                      If True, training status is printed.
        """

        # No gradients are needed to train ELLA
        with torch.no_grad():
            # Get numpy random element
            rng = np.random.default_rng(self.seed)
            # Get indexes for the elements used to build the kernel approximation.
            indexes = rng.choice(np.arange(X_train.shape[0]), self.M, replace=False)
            # Get training instances
            Xz = X_train[indexes]

            # Compute the Jacobians of those inputs on their corresponding
            #  target value.
            # phi_z shape (self.M, 1, n_parameters)
            if self.output_size > 1:
                phi_z = self.net.get_jacobian_on_outputs(
                    Xz, y_train[indexes].to(torch.long).squeeze(-1)
                )
            else:
                phi_z = self.net.get_jacobian(Xz)

            # Squeeze output dimension (1).
            # phi_z shape is (self.M, n_params)
            phi_z = phi_z.squeeze(1)

            # Compute the estimated kernel as phi @ phi.T
            #  shape (self.M, self.M)
            K = torch.einsum("ns, ms -> nm", phi_z, phi_z)

            # Compute eigen-vectors and eigen-values
            L, V = torch.linalg.eigh(K)

            # Get self.K highest eigen values and their eigen-vectors.
            # correct negative values (due to computation errors)
            # and reorder in decreasing order.
            L = torch.abs(L[-self.K :]).flip(-1)
            V = V[:, -self.K :].flip(-1)

            # Compute embedding vector v as phi @ V/sqrt(L)
            self.v = torch.einsum("ms, mk -> sk", phi_z, V / torch.sqrt(L).unsqueeze(0))

            # Initialize G matrix
            self.G = torch.zeros(self.K, self.K, dtype=self.dtype)

            # Initialize loader with verbose or not
            if verbose:
                from tqdm import tqdm

                iters = tqdm(range(len(train_loader)), unit="iteration")
                iters.set_description("Training ")
            else:
                iters = np.arange(len(train_loader))

            data_iter = iter(train_loader)

            # Initialize validation NLL
            best_vall_nll = np.infty

            # Batches iteration
            for i in iters:
                # If evaluation iteration
                if i % val_steps == 0 and val_loader is not None:
                    # Compute validation NLL
                    val_nll = score(
                        self,
                        val_loader,
                        self.val_metrics,
                        use_tqdm=verbose,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    # If improved, store new best value
                    if val_nll < best_vall_nll:
                        best_vall_nll = val_nll
                    # Else, training has ended
                    else:
                        print("Training ended at {} iterations".format(i))
                        break

                # Get batch of data
                data, target = next(data_iter)
                data = data.to(self.device).to(self.dtype)
                target = target.to(self.device).to(self.dtype)

                # Compute likelihood function hessian
                Lambda = self.likelihood_hessian(data, target)

                # Compute jacobian of the batch
                Jtrain = self.net.get_jacobian(data)
                # Compute low-rank feature vector as J@v
                # n is batch dimension
                # d is output dimension
                # s is parameter dimension
                # k is low-rank dimension
                phi_train = torch.einsum("nds, sk -> ndk", Jtrain, self.v)
                # Add contribution to G
                # n is batch dimension
                # a and b are output dimension
                # k and g are low-rank dimension
                self.G += torch.einsum(
                    "nak, abn, nbg -> kg", phi_train, Lambda, phi_train
                )

    def predict_y(self, predict_at):
        """Computes the predictions of the model."""
        raise NotImplementedError

    def forward(self, predict_at):
        """
        Computes the predicted target values for the given input.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.

        Returns
        -------
        The predicted mean and variance.
        """
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        mean, var = self.predict_y(predict_at)
        # Return predictions scaled to the original scale.
        return mean * self.y_std + self.y_mean, var * self.y_std**2


class ELLA_Regression(ELLA_Base):
    """Initialices a ELLA model for regression in 1D.

    Parameters
    ----------
        net : callable
              Deterministic deep learning moden in which to a apply LLA. Must
              have methods "jacobians_on_outputs" to compute its jacobians.
        output_size : int
                      Number of output dimenions of the deep learning model.
        n_samples : int
                    Amount of MC samples to take from the training dataset
                    to approximate the kernel matrix.
        n_eigh : int
                 Number of eigenvalues to consider for the features
                 approximation.
        prior_std : float
                    Parameters' prior standard deviation of their isotropic
                    Gaussian distribution.
        seed : int
               Random seed to ensure reproducilibity of the code.
        log_variance : float
                       noise variance for gaussian likelihood.
        y_mean : float
                 Mean of the target values, used to re-escale outputs.
        y_std : float
                Standard deviation of the target_values, used to re-escale oututs.
        device : torch.device
                 Device in which to perform inference.
        dtype : torch.dtype
                Precision used for the model.
    """

    def __init__(
        self,
        net,
        output_size,
        n_samples,
        n_eigh,
        prior_std,
        seed,
        log_variance,
        y_mean,
        y_std,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            output_size,
            n_samples,
            n_eigh,
            prior_std,
            seed,
            y_mean,
            y_std,
            device=None,
            dtype=torch.float64,
        )
        # Create tensor with noise variance
        self.log_variance = torch.tensor(log_variance, device=device, dtype=dtype)
        # Store the regression metric for validation
        self.val_metrics = RegressionNLL

    def likelihood_hessian(self, data, target):
        """Computes the hessian of the likelihood function wrt the model output.
        In regression, this is an identity matrix divided by the likelihood
        noise variance.
        """
        # Get identity matrix
        ones = torch.ones_like(target).unsqueeze(-1).permute(1, 2, 0)
        # Divide it by noise variance.
        return ones / self.log_variance.exp()

    def predict_y(self, X):
        mean, var = self.predict_f(X)
        return mean, var + self.log_variance.exp()


class ELLA_Multiclass(ELLA_Base):
    def __init__(
        self,
        net,
        output_size,
        n_samples,
        n_eigh,
        prior_std,
        seed,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            output_size,
            n_samples,
            n_eigh,
            prior_std,
            seed,
            0,
            1,
            device=device,
            dtype=dtype,
        )
        self.val_metrics = SoftmaxClassificationNLL

    def likelihood_hessian(self, data, target):
        """Computes the hessian of the likelihood function wrt the model output.
        In classification, this is:

        diag(probs) - probs.T @ probs

        """
        out = torch.nn.Softmax(dim=-1)(self.net(data))
        a = torch.einsum("na, nb -> abn", out, out)
        b = torch.diag_embed(out).permute(1, 2, 0)
        return -a + b

    def predict_y(self, X):
        return self.predict_f(X)


class ELLA_MulticlassBackpack(ELLA_Multiclass):
    """Extends ELLA_multiclass to use BackPack in order to compute Jacobians."""

    def fit_loader_val(
        self, X_train, y_train, train_loader, val_loader, val_steps=1, verbose=True
    ):
        """Fits ELLA model to the given training dataset. Performs early-stopping using
        a validation dataset.

        Parametes
        ---------
            X_train : torch.tensor of shape (n_train, input_shape)
                      Contains the input dataset.
            y_train : torch.tensor of shape (n_train, )
                      Contains the input labels.
            train_loader : torch.datasets.loader
                           Dataloader instance of the same training dataset
                           provided on (X_train, y_train).
            val_loader : torch.datasets.loader
                        Dataloader instance of a validation dataset.
            val_steps : int
                        Number of iterations between validations.
            verbose : boolean
                      If True, training status is printed.
        """

        # No gradients are needed to train ELLA
        with torch.no_grad():
            # Get numpy random element
            rng = np.random.default_rng(self.seed)
            # Get indexes for the elements used to build the kernel approximation.
            indexes = rng.choice(np.arange(X_train.shape[0]), self.M, replace=False)
            # Get training instances
            Xz = X_train[indexes]

            # Compute the Jacobians of those inputs on their corresponding
            #  target value.
            # phi_z shape (self.M, 1, n_parameters)
            if self.output_size > 1:
                phi_z = self.backpack.jacobians_on_outputs(
                    Xz, y_train[indexes].to(torch.long), enable_back_prop=False
                )
            else:
                phi_z = self.backpack.jacobians(Xz, enable_back_prop=False)

            # Squeeze output dimension (1).
            # phi_z shape is (self.M, n_params)
            phi_z = phi_z.squeeze(1)

            # Compute the estimated kernel as phi @ phi.T
            #  shape (self.M, self.M)
            K = torch.einsum("ns, ms -> nm", phi_z, phi_z)

            # Compute eigen-vectors and eigen-values
            L, V = torch.linalg.eigh(K)

            # Get self.K highest eigen values and their eigen-vectors.
            # correct negative values (due to computation errors)
            # and reorder in decreasing order.
            L = torch.abs(L[-self.K :]).flip(-1)
            V = V[:, -self.K :].flip(-1)

            # Compute embedding vector v as phi @ V/sqrt(L)
            self.v = torch.einsum("ms, mk -> sk", phi_z, V / torch.sqrt(L).unsqueeze(0))

            # Initialize G matrix
            self.G = torch.zeros(self.K, self.K, dtype=self.dtype)

            # Initialize loader with verbose or not
            if verbose:
                from tqdm import tqdm

                iters = tqdm(range(len(train_loader)), unit="iteration")
                iters.set_description("Training ")
            else:
                iters = np.arange(len(train_loader))

            data_iter = iter(train_loader)

            # Initialize validation NLL
            best_vall_nll = np.infty

            # Batches iteration
            for i in iters:
                # If evaluation iteration
                if i % val_steps == 0 and val_loader is not None:
                    # Compute validation NLL
                    val_nll = score(
                        self,
                        val_loader,
                        self.val_metrics,
                        use_tqdm=verbose,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    # If improved, store new best value
                    if val_nll < best_vall_nll:
                        best_vall_nll = val_nll
                    # Else, training has ended
                    else:
                        print("Training ended at {} iterations".format(i))
                        break

                # Get batch of data
                data, target = next(data_iter)
                data = data.to(self.device).to(self.dtype)
                target = target.to(self.device).to(self.dtype)

                # Compute likelihood function hessian
                Lambda = self.likelihood_hessian(data, target)

                # Compute jacobian of the batch
                Jtrain = self.backpack.jacobians(data, enable_back_prop=False)
                # Compute low-rank feature vector as J@v
                # n is batch dimension
                # d is output dimension
                # s is parameter dimension
                # k is low-rank dimension
                phi_train = torch.einsum("nds, sk -> ndk", Jtrain, self.v)
                # Add contribution to G
                # n is batch dimension
                # a and b are output dimension
                # k and g are low-rank dimension
                self.G += torch.einsum(
                    "nak, abn, nbg -> kg", phi_train, Lambda, phi_train
                )

    def predict_f(self, X):
        """
        Defines the predictive computation of ELLA.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        Returns
        -------
        mean : torch tensor of size (batch_size, output_dim)
               Predictive mean of the model on the given batch
        K : torch tensor of size (batch_size, output_dim, output_dim)
            Contains the covariance matrix of the model for each element on
            the batch.
        """

        # No grad is required for ELLA computations
        with torch.no_grad():
            # Compute G inverse adding the prior.
            prior = torch.eye(self.K, device=self.device, dtype=self.dtype) / (
                self.prior_std**2
            )
            G_inv = torch.inverse(self.G + prior)

            # Get Jacobians with shape (batch_size, output_size, n_parameters)
            J = self.backpack.jacobians(X, enable_back_prop=False)
            # Compute low-dimensional features as J @ v
            # self.v has shape (n_parameters, self.n_eigh)
            # low-dimensional features are (batch_size, output_dim, self.n_eigh)
            phi = torch.einsum("mds, sk -> mdk", J, self.v)

            # Covariance matrix is computed as phi @ G^{-1} @ phi.T
            # n is batch_size
            # a and b is output dimension
            # k and g are self.n_eigh dimension
            K = torch.einsum("nak, kg, nbg -> nab", phi, G_inv, phi)

            # Compute mean using network
            mean = self.net(X)

        return mean, K
