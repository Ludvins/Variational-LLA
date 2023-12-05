import numpy as np
import torch
from tqdm import tqdm



def fit_map(
    model,
    training_generator,
    optimizer,
    criterion,
    iterations=None,
    use_tqdm=False,
    return_loss=False,
    device=None,
    ):
    # Set model in training mode

    losses = []

    model.train()

    if use_tqdm:
        # Initialize TQDM bar
        iters = tqdm(range(iterations), unit=" iteration")
        iters.set_description("Training ")
    else:
        iters = range(iterations)
    data_iter = iter(training_generator)

    for _ in iters:
        try:
            inputs, target = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iter = iter(training_generator)
            inputs, target = next(data_iter)
        inputs = inputs.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()   # zero the gradient buffers
        output = model(inputs)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if return_loss:
            losses.append(loss.detach().cpu().numpy())

    return losses

def fit_map_crossentropy(
    model,
    training_generator,
    optimizer,
    criterion,
    iterations=None,
    use_tqdm=False,
    return_loss=False,
    device=None,
    dtype = None
):
    # Set model in training mode

    losses = []

    model.train()

    if use_tqdm:
        # Initialize TQDM bar
        iters = tqdm(range(iterations), unit=" iteration")
        iters.set_description("Training ")
    else:
        iters = range(iterations)
    data_iter = iter(training_generator)

    for _ in iters:
        try:
            inputs, target = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iter = iter(training_generator)
            inputs, target = next(data_iter)
        inputs = inputs.to(device).to(dtype)
        target = target.to(device).to(dtype)
        
        optimizer.zero_grad()   # zero the gradient buffers
        output = model(inputs)

        loss = criterion(output, target.squeeze(-1).to(torch.long))
        loss.backward()
        optimizer.step()
        if return_loss:
            losses.append(loss.detach().cpu().numpy())

    return losses

def acc_multiclass(
    model,
    generator,
    use_tqdm=False,
    device=None,
    dtype = None
):
    # Set model in training mode


    model.eval()

    if use_tqdm:
        # Initialize TQDM bar
        iters = tqdm(range(len(generator)), unit=" iteration")
        iters.set_description("Evaluating ")
    else:
        iters = len(generator)
    data_iter = iter(generator)
    
    correct = 0
    total = 0
    
    for _ in iters:
        inputs, target = next(data_iter)
        inputs = inputs.to(device).to(dtype)
        target = target.to(device).to(dtype)
        
        output = model(inputs)
        
        prediction = torch.argmax(output, -1).unsqueeze(-1)

        correct += torch.sum(prediction == target)
    
        total += prediction.shape[0]

    return (correct/total).detach().cpu().numpy()


def fit(
    model,
    training_generator,
    optimizer,
    val_generator = None,
    val_steps = -1,
    val_metrics = None,
    iterations=None,
    use_tqdm=False,
    return_loss=False,
    device=None,
    dtype=None,
):
    """
    Trains the given model using the arguments provided.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    training_generator : iterable
                         Must return batches of pairs corresponding to the
                         given inputs and target values.
    optimizer : torch optimizer
                The already initialized optimizer.
    scheduler : torch scheduler
                Learning rate scheduler.
    epochs : int
             Number of epochs to train de model.
    device : torch device
             Device in which to perform all computations.
    """
    # Set model in training mode

    losses = []
    val_losses = [np.infty]

    #model.train()

    if use_tqdm:
        # Initialize TQDM bar
        iters = tqdm(range(iterations), unit=" iteration")
        iters.set_description("Training ")
    else:
        iters = range(iterations)
    data_iter = iter(training_generator)

    for i in iters:
        try:
            inputs, target = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iter = iter(training_generator)
            inputs, target = next(data_iter)
        inputs = inputs.to(device)
        target = target.to(device)
        loss = model.train_step(optimizer, inputs, target)
        if return_loss:
            losses.append(loss.detach().cpu().numpy())

        if val_generator != None and i%val_steps == 0:
            #model.eval()
            sc = score(model, val_generator, val_metrics, use_tqdm, device, dtype)["NLL"]
            #print("Validation NLL: {}".format(sc))
            val_losses.append(sc)
            if val_losses[-1]>val_losses[-2]:
                #print("Early-stopped at iteration {} with val NLL {}".format(i, val_losses[-1]))
                return losses, val_losses[1:]
            #model.train()

    return losses, val_losses[1:]



def fit(
    model,
    training_generator,
    optimizer,
    val_generator = None,
    val_steps = -1,
    val_metrics = None,
    iterations=None,
    use_tqdm=False,
    return_loss=False,
    device=None,
    dtype=None,
):
    """
    Trains the given model using the arguments provided.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    training_generator : iterable
                         Must return batches of pairs corresponding to the
                         given inputs and target values.
    optimizer : torch optimizer
                The already initialized optimizer.
    scheduler : torch scheduler
                Learning rate scheduler.
    epochs : int
             Number of epochs to train de model.
    device : torch device
             Device in which to perform all computations.
    """
    # Set model in training mode

    losses = []
    val_losses = [np.infty]

    #model.train()

    if use_tqdm:
        # Initialize TQDM bar
        iters = tqdm(range(iterations), unit=" iteration")
        iters.set_description("Training ")
    else:
        iters = range(iterations)
    data_iter = iter(training_generator)

    for i in iters:
        try:
            inputs, target = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iter = iter(training_generator)
            inputs, target = next(data_iter)
        inputs = inputs.to(device)
        target = target.to(device)
        loss = model.train_step(optimizer, inputs, target)
        if return_loss:
            losses.append(loss.detach().cpu().numpy())

        if val_generator != None and i%val_steps == 0:
            #model.eval()
            sc = score(model, val_generator, val_metrics, use_tqdm, device, dtype)["NLL"]
            print("Iteration ", i)
            print("Validation NLL: {}".format(sc))
            val_losses.append(sc)
            if val_losses[-1]>val_losses[-2]:
                print("Early-stopped at iteration {} with val NLL {}".format(i, val_losses[-1]))
                return losses, val_losses[1:]
            #model.train()

    return losses, val_losses[1:]


def score(model, generator, metrics, use_tqdm=False, device=None, dtype = None, **kwargs):
    """
    Evaluates the given model using the arguments provided.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    generator : iterable
                Must return batches of pairs corresponding to the
                given inputs and target values.
    device : torch device
             Device in which to perform all computations.

    Returns
    -------
    metrics : dictionary
              Contains pairs of (metric, value) averaged over the number of
              batches.
    """
    # Set model in evaluation mode
    # Initialize metrics

    metrics = metrics(len(generator.dataset), device=device, dtype = dtype)

    if use_tqdm:
        # Initialize TQDM bar
        iters = tqdm(range(len(generator)), unit="iteration")
        iters.set_description("Evaluating ")
    else:
        iters = range(len(generator))
    data_iter = iter(generator)

    with torch.no_grad():
        # Batches evaluation
        for _ in iters:
            data, target = next(data_iter)
            data = data.to(device)
            target = target.to(device)
            loss, Fmean, Fvar = model.test_step(data, target)
            # log_likelihood = model.predict_logdensity(data, target)
            # Update mertics using this batch
            metrics.update(
                target, loss, Fmean, Fvar
            )
    # Return metrics as a dictionary
    d = metrics.get_dict()
    metrics.reset()
    return d


def predict(model, generator, device=None):
    """
    Creates the model predictions for the given data generator.
    The model predictive distribution is a Gaussian mixture.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    generator : iterable
                Must return batches of pairs corresponding to the
                given inputs and target values.
    device : torch device
             Device in which to perform all computations.

    Returns
    -------
    means : numpy darray of shape (S, N, D)
            Contains the predicted mean for every mixture (S),
            datapoint (N) with the corresponding output dimension (D).
    vars : numpy darray of shape (S, N, D)
           Contains the predicted variance of every mixture (S),
           datapoint (N), with the corresponding output dimension (D).
    """

    # Create containers
    means, vars = [], []

    with torch.no_grad():
        # Look over batches of data.
        for idx, data in enumerate(generator):
            # Consider datasets with no targets, just input values.
            try:
                batch_x, _ = data
            except:
                batch_x = data
            # Create batched predictions
            batch_means, batch_vars = model.predict_mean_and_var(batch_x.to(device))
            # Apped to the arrays
            means.append(batch_means.detach().cpu().numpy())
            vars.append(batch_vars.detach().cpu().numpy())

    # Concatenate batches on second dimension, the first
    # corresponds to the mixture.
    means = np.concatenate(means, axis=0)
    vars = np.concatenate(vars, axis=0)

    return means, vars




def forward(model, generator, device=None):
    """
    Creates the model predictions for the given data generator.
    The model predictive distribution is a Gaussian mixture.

    Arguments
    ---------
    model : torch.nn.Module
            Torch model to train.
    generator : iterable
                Must return batches of pairs corresponding to the
                given inputs and target values.
    device : torch device
             Device in which to perform all computations.

    Returns
    -------
    means : numpy darray of shape (S, N, D)
            Contains the predicted mean for every mixture (S),
            datapoint (N) with the corresponding output dimension (D).
    vars : numpy darray of shape (S, N, D)
           Contains the predicted variance of every mixture (S),
           datapoint (N), with the corresponding output dimension (D).
    """

    # Create containers
    means, vars = [], []

    # Look over batches of data.
    for idx, data in enumerate(generator):
        # Consider datasets with no targets, just input values.
        try:
            batch_x, _ = data
        except:
            batch_x = data
        # Create batched predictions
        batch_means, batch_vars = model(batch_x.to(device))
        # Apped to the arrays
        means.append(batch_means.detach().cpu().numpy())
        vars.append(batch_vars.detach().cpu().numpy())

    # Concatenate batches on second dimension, the first
    # corresponds to the mixture.
    means = np.concatenate(means, axis=0)
    vars = np.concatenate(vars, axis=0)

    return means, vars
