import torch
import numpy as np


class BaseVaLLA(torch.nn.Module):
    """
    Defines a Sparse Linearized Laplace Approximation model.

    Parameters
    ----------
    net_forward : Callable
                  Forward method of the deep model on which LLA is being applied.
    Z : array of size (num_inducing, input_dim)
        Contains the inducing locations of the model.
    prior_std : float
                Value of the standrd deviation of the Gaussian prior over parameters.
    likelihood : Likelihood
                 Indicates the likelihood distribution of the data.
    num_data : int
                Amount of data samples in the full dataset. This is used
                to scale the likelihood in the loss function to the size
                of the minibatch.
    output_dim : int
                 Dimensionality of the targets.
    backend : Callable
              Returns the Jacobian of the deep model with respect to the given input.
    track_inducing_locations : Boolean
                               If True, an history of the inducing locations is stored.
    fix_inducing_locations : Boolean
                             If True, the inducing locations are fixed to their initial
                             value.
    alpha : float
            Alpha value used for BlackBox alpha energy learning.
            When 0, the usual ELBO from variational inference is used.
    y_mean : float or array-like
                The given target values at training must be normalized.
                This variable indicates the original mean value so that
                the computed metrics follow the original scale.
    y_std : float or array-like
            Original standar deviation of the normalized targets.
    device : torch.device
                The device in which the computations are made.
    dtype : data-type
            The dtype of the layer's computations and weights.
    """

    def __init__(
        self,
        net,
        Z,
        prior_std,
        num_data,
        output_dim,
        track_inducing_locations=False,
        trainable_prior=True,
        inducing_classes=None,
        y_mean=0.0,
        y_std=1.0,
        alpha=1.0,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__()
        # Store data information
        self.num_data = num_data
        self.output_dim = output_dim
        # Store targets mean and std.
        self.y_mean = torch.tensor(y_mean, device=device)
        self.y_std = torch.tensor(y_std, device=device)

        self.alpha = alpha
        self.net = net
        self.num_inducing = Z.shape[0]

        self.log_prior_std = torch.log(
            torch.tensor(prior_std, device=device, dtype=dtype)
        )
        if trainable_prior:
            self.log_prior_std = torch.nn.Parameter(self.log_prior_std)

        self.inducing_locations = torch.tensor(Z, device=device, dtype=dtype)
        self.inducing_locations = torch.nn.Parameter(self.inducing_locations)

        if inducing_classes is None:
            print(
                "Initializing inducing classes with {} different classes".format(
                    self.output_dim
                )
            )
            self.inducing_class = torch.tensor(
                np.tile(
                    np.arange(self.output_dim),
                    reps=np.ceil(Z.shape[0] / self.output_dim).astype(int),
                )[: Z.shape[0]],
                device=device,
                dtype=torch.long,
            )
        else:
            self.inducing_class = torch.tensor(
                inducing_classes, device=device, dtype=torch.long
            )

        self.track_inducing_locations = track_inducing_locations
        if track_inducing_locations:
            self.inducing_history = [
                self.inducing_locations.clone().detach().cpu().numpy()
            ]

        # Set device and data type (precision)
        self.device = device
        self.dtype = dtype

        # Initialize cholesky decomposition of identity
        I = np.eye(self.num_inducing)

        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        triangular_q_sqrt = I[li, lj]
        self.L = torch.tensor(
            triangular_q_sqrt,
            dtype=self.dtype,
            device=self.device,
        )

        self.L = torch.nn.Parameter(self.L)
        self.ell_history = []
        self.kl_history = []

    def train_step(self, optimizer, X, y):
        """
        Defines the training step for the DVIP model using a simple optimizer.
        This method illustrates a standard training step. If more complex
        operations are needed, such as optimizers with double steps,
        create your own training step, calling this one is not compulsory.
        Parameters
        ----------
        optimizer : torch.optim
                    The considered optimization algorithm.
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input, must be standardized.
        Returns
        -------
        loss : float
               The nelbo of the model at the current state for the
               given inputs.
        """
        self.train()
        # If targets are unidimensional,
        # ensure there is a second dimension (N, 1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Transform inputs and largets to the model'd dtype
        if self.dtype != X.dtype:
            X = X.to(self.dtype)
        if self.dtype != y.dtype:
            y = y.to(self.dtype)

        # Clear gradients
        optimizer.zero_grad()

        # Compute loss
        loss = self.nelbo(X, y)

        # Create backpropagation graph
        loss.backward()

        # Make optimization step
        optimizer.step()

        if self.track_inducing_locations:
            self.inducing_history += [
                self.inducing_locations.clone().detach().cpu().numpy()
            ]

        return loss

    def test_step(self, X, y):
        """
        Defines the test step for the DVIP model.
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
        mean_pred : torch tensor of size (batch_size, output_dim)
                    Predictive mean of the model on the given batch
        var_pred : torch tensor of size (batch_size, output_dim, output_dim)
                   Contains the covariance matrix of the model for each element on
                   the batch.
        """

        self.eval()
        with torch.no_grad():
            # In case targets are one-dimensional and flattened, add a final dimension.
            if y.ndim == 1:
                y = y.unsqueeze(-1)

            # Cast types if needed.
            if self.dtype != X.dtype:
                X = X.to(self.dtype)
            if self.dtype != y.dtype:
                y = y.to(self.dtype)

            Fmean, Fvar = self(X)  # Forward pass

            # Temporarily change the num data variable so that the
            # scale of the likelihood is correctly computed on the
            # test dataset.
            num_data = self.num_data
            self.num_data = X.shape[0]
            # Compute the loss with scaled data
            loss = self.nelbo(X, (y - self.y_mean) / self.y_std)
            self.num_data = num_data

            return loss, Fmean, Fvar

    def compute_inducing_term(self, Kz):
        # Transform flattened cholesky decomposition parameter into matrix
        L = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        # Shape (num_inducing, num_inducing)
        L[li, lj] = self.L

        # Compute auxiliar matrices
        # Shape [num_inducing, num_inducing]
        # H = I + L^T @ self.Kz @ L
        I = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)
        self.H = I + torch.einsum("mn, ml, lk -> nk", L, Kz, L)
        # Shape [num_inducing, num_inducing]
        # A = L @ H^{-1} @ L^T
        # self.A = torch.einsum("nm, ml, kl -> nk", L, torch.inverse(self.H), L)
        self.A = L @ torch.linalg.solve(self.H, L.T)

    def compute_kernels(self, X):
        _, Kx_diag, Kxz, Kz = self.net.get_full_kernels(
            X,
            self.inducing_locations,
            self.inducing_class,
        )
        Kx_diag = torch.exp(self.log_prior_std) ** 2 * Kx_diag
        Kxz = torch.exp(self.log_prior_std) ** 2 * Kxz
        Kz = torch.exp(self.log_prior_std) ** 2 * Kz
        F_mean = self.net(X)
        return F_mean, Kx_diag, Kxz, Kz

    def predict_f(self, X):
        """
        Performs the mean and covariance matrix of the given input.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.

        Returns
        -------
        mean_pred : torch tensor of size (batch_size, output_dim)
                    Predictive mean of the model on the given batch
        var_pred : torch tensor of size (batch_size, output_dim, output_dim)
                   Contains the covariance matrix of the model for each element on
                   the batch.
        """

        # Transform flattened cholesky decomposition parameter into matrix
        L = torch.eye(self.num_inducing, dtype=self.dtype, device=self.device)
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing)
        # Shape (num_inducing, num_inducing)
        L[li, lj] = self.L

        F_mean, Kx_diag, Kxz, Kz = self.compute_kernels(X)
        self.Kz = Kz
        self.compute_inducing_term(Kz)

        # Compute predictive diagonal
        # Shape [output_dim, output_dim, batch_size, batch_size]
        # K2 = Kxz @ A @ Kxz^T
        diag = torch.einsum("nma, ml, nlb -> nab", Kxz, self.A, Kxz)
        # Shape [batch_size, output_dim, output_dim]
        Fvar = Kx_diag - diag
        return F_mean, Fvar

    def forward(self, X):
        raise NotImplementedError

    def compute_KL(self):
        """
        Computes the Kulback-Leibler divergence between the variational distribution
        and the prior.
        """

        log_det = torch.logdet(self.H)
        trace = torch.sum(torch.diagonal(self.Kz @ self.A))
        KL = 0.5 * log_det - 0.5 * trace
        return torch.sum(KL)

    def nelbo(self, X, y):
        raise NotImplementedError

    def print_variables(self):
        """Prints the model variables in a prettier format."""
        import numpy as np

        print("\n---- MODEL PARAMETERS ----")
        np.set_printoptions(threshold=3, edgeitems=2)
        sections = []
        pad = "  "
        for name, param in self.named_parameters():
            if param.requires_grad == False:
                continue
            name = name.split(".")
            for i in range(len(name) - 1):
                if name[i] not in sections:
                    print(pad * i, name[i].upper())
                    sections = name[: i + 1]

            padding = pad * (len(name) - 1)
            print(
                padding,
                "{}: ({})".format(name[-1], str(list(param.data.size()))[1:-1]),
            )
            print(
                padding + " " * (len(name[-1]) + 2),
                param.data.detach().cpu().numpy().flatten(),
            )

        print("\n---------------------------\n\n")

    def freeze_cholesky(self):
        self.L.requires_grad = False

    def freeze_inducing(self):
        self.inducing_locations.requires_grad = False

    def freeze_prior(self):
        self.log_prior_std.requires_grad = False


class VaLLARegression(BaseVaLLA):
    def __init__(
        self,
        net,
        Z,
        prior_std,
        num_data,
        output_dim,
        log_variance=-5,
        track_inducing_locations=False,
        trainable_prior=True,
        inducing_classes=None,
        y_mean=0.0,
        y_std=1.0,
        alpha=1.0,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            Z,
            prior_std,
            num_data,
            output_dim,
            track_inducing_locations,
            trainable_prior,
            inducing_classes,
            y_mean,
            y_std,
            alpha,
            device,
            dtype,
        )

        self.log_variance = torch.tensor(log_variance, device=device, dtype=dtype)
        self.log_variance = torch.nn.Parameter(self.log_variance)

    def freeze_ll(self):
        self.log_variance.requires_grad = False

    def forward(self, predict_at):
        """
        Computes the predicted labels for the given input.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.

        Returns
        -------
        The predicted mean and standard deviation.
        """
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        mean, var = self.predict_y(predict_at)
        # Return predictions scaled to the original scale.
        return mean * self.y_std + self.y_mean, var * self.y_std**2

    def predict_y(self, X):
        mean, var = self.predict_f(X)
        return mean, var + self.log_variance.exp()

    def logdensity(self, mu, var, x):
        """Computes the log density of a one dimensional Gaussian distribution
        of mean mu and variance var, evaluated on x.
        """
        logp = -0.5 * (np.log(2 * np.pi) + torch.log(var) + (mu - x) ** 2 / var)
        return logp

    def nelbo(self, X, y):
        """
        Computes the negative ELBO in the Hilbert space.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.
        y : torch tensor of shape (batch_size, output_dim)
            Targets of the given input.

        Returns
        -------
        elbo : float
               The nelbo of the model at the current state for the given inputs.
        """
        F_mean, F_var = self.predict_f(X)
        if self.alpha == 0:
            logpdf = (
                -0.5 * np.log(2 * np.pi)
                - 0.5 * self.log_variance
                - 0.5
                * ((y - F_mean) ** 2 + F_var.squeeze(-1))
                / self.log_variance.exp()
            )
            logpdf = torch.sum(logpdf, -1)
        else:
            # Black-box alpha-energy
            variance = torch.exp(self.log_variance)
            # Proportionality constant
            C = (
                torch.sqrt(2 * torch.pi * variance / self.alpha)
                / torch.sqrt(2 * torch.pi * variance) ** self.alpha
            )

            logpdf = self.logdensity(
                F_mean, F_var.squeeze(-1) + variance / self.alpha, y
            )
            logpdf = logpdf + torch.log(C)
            logpdf = logpdf / self.alpha

        # # Aggregate on data dimension
        logpdf = torch.sum(logpdf)
        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = self.compute_KL()

        self.ell_history.append((-scale * logpdf).detach().cpu().numpy())
        self.kl_history.append(KL.detach().cpu().numpy())

        return -scale * logpdf + KL

    def predict_mean_and_var(self, X):
        """
        Computes the Predictive mean and variance of the model using the likelihood.

        Parameters
        ----------
        X : torch tensor of shape (batch_size, data_dim)
            Contains the input features.

        Returns
        -------
        mean_pred : torch tensor of size (batch_size, output_dim)
                    Predictive mean of the model on the given batch
        var_pred : torch tensor of size (batch_size, output_dim, output_dim)
                   Contains the covariance matrix of the model for each element on
                   the batch.
        """
        Fmu, F_var = self(X)
        return Fmu, F_var + self.log_variance.exp()


class VaLLARegressionBackend(VaLLARegression):
    def __init__(
        self,
        net,
        Z,
        prior_std,
        num_data,
        output_dim,
        log_variance,
        backend,
        track_inducing_locations=False,
        trainable_prior=True,
        alpha=1.0,
        inducing_classes=None,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            Z,
            prior_std,
            num_data,
            output_dim,
            log_variance,
            track_inducing_locations,
            trainable_prior,
            inducing_classes,
            y_mean,
            y_std,
            alpha,
            device,
            dtype,
        )
        self.backpack = backend

    def compute_kernels(self, X):
        Jx = self.backpack.jacobians(X, enable_back_prop=False)
        Jz = self.backpack.jacobians_on_outputs(
            self.inducing_locations,
            self.inducing_class.unsqueeze(-1),
            enable_back_prop=self.training,
        ).squeeze(1)

        Kx_diag = torch.exp(self.log_prior_std) ** 2 * torch.einsum(
            "nai, nbi -> nab", Jx, Jx
        )
        Kxz = torch.exp(self.log_prior_std) ** 2 * torch.einsum(
            "nai, mi -> nma", Jx, Jz
        )
        Kzz = torch.exp(self.log_prior_std) ** 2 * torch.einsum("ni, mi -> nm", Jz, Jz)
        F_mean = self.net(X)

        return F_mean, Kx_diag, Kxz, Kzz


class VaLLAMultiClass(BaseVaLLA):
    def nelbo(self, X, y):
        F_mean, F_var = self.predict_f(X)

        F = F_mean / torch.sqrt(
            1 + torch.pi / 8 * torch.diagonal(F_var, dim1=1, dim2=2)
        )

        probs = F.softmax(-1) ** self.alpha

        ell = (
            -1
            / self.alpha
            * torch.nn.functional.cross_entropy(
                probs.log(), y.to(torch.long).squeeze(-1), reduction="none"
            )
        )

        # # Aggregate on data dimension
        logpdf = torch.sum(ell)
        # Scale loss term corresponding to minibatch size
        scale = self.num_data
        scale /= X.shape[0]

        # Compute KL term
        KL = self.compute_KL()

        self.ell_history.append((-scale * logpdf).detach().cpu().numpy())
        self.kl_history.append(KL.detach().cpu().numpy())

        return -scale * logpdf + KL

    def forward(self, predict_at):
        """
        Computes the predicted labels for the given input.

        Parameters
        ----------
        predict_at : torch tensor of shape (batch_size, data_dim)
                     Contains the input features.

        Returns
        -------
        The predicted mean and standard deviation.
        """
        # Cast types if needed.
        if self.dtype != predict_at.dtype:
            predict_at = predict_at.to(self.dtype)

        if self.y_mean != 0 or self.y_std != 1:
            raise ValueError("Labels should not be standardized in Classification")

        return self.predict_f(predict_at)



class VaLLAMultiClassBackend(VaLLAMultiClass):
    def __init__(
        self,
        net,
        Z,
        prior_std,
        num_data,
        output_dim,
        backend,
        track_inducing_locations=False,
        trainable_prior=True,
        alpha=1.0,
        inducing_classes=None,
        y_mean=0.0,
        y_std=1.0,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(
            net,
            Z,
            prior_std,
            num_data,
            output_dim,
            track_inducing_locations,
            trainable_prior,
            inducing_classes,
            y_mean,
            y_std,
            alpha,
            device,
            dtype,
        )
        self.backpack = backend

    def compute_kernels(self, X):
        Jx = self.backpack.jacobians(X, enable_back_prop=False)
        Jz = self.backpack.jacobians_on_outputs(
            self.inducing_locations,
            self.inducing_class.unsqueeze(-1),
            enable_back_prop=self.training,
        ).squeeze(1)

        Kx_diag = torch.exp(self.log_prior_std) ** 2 * torch.einsum(
            "nai, nbi -> nab", Jx, Jx
        )
        Kxz = torch.exp(self.log_prior_std) ** 2 * torch.einsum(
            "nai, mi -> nma", Jx, Jz
        )
        Kzz = torch.exp(self.log_prior_std) ** 2 * torch.einsum("ni, mi -> nm", Jz, Jz)
        F_mean = self.net(X)

        return F_mean, Kx_diag, Kxz, Kzz
