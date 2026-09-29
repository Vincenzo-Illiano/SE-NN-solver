from data import generators

def trainFno1d(model, trainingset, optimizer, epochs, loss, device="cuda", printprogress = False, shuffledataset = False):
    """
    Trains a Fno1d model and returns the trained model and the average training losses for each epoch.
    
    Arguments:
    model -- model to train
    trainingset -- training set 
    optimizer -- optimizer used in the training
    epochs -- number of epochs to be used in the training
    loss -- loss function needed. This must be a function like loss(E, phi, Hphi), where phi and Hphi are
        functions with N points representing the wavefunction and the hamiltonian applied to the wavefunction
    device -- device used for training. Can be "cuda" or "cpu". (default cuda)
    printprogress -- whether to print the training progress
    shuffledataset -- whether to shuffle the dataset before each epoch
    """

    # Moves the model to the used device
    model.to(device)

    # Losses for each epoch
    train_losses = []

    # Gets information about the training set
    trainset = trainingset['function']
    num_batches = trainingset['num_batches']
    N = trainingset['N']

    # Makes sure that the functions that the model takes have the same size as the ones in the dataset
    if(N != model.N):
        raise ValueError(f"Model expects function size {model.N}, but trainingset has function size {N}")

    # Makes sure to be in training mode
    model.train()

    # Does the training
    for j in range(epochs):

        # Shuffles the dataset
        if(shuffledataset):
            generators.shuffleDataset(trainset)

        running_loss = 0

        for i, batch in enumerate(trainset):
            optimizer.zero_grad()

            # Moves the batches to the used device
            batch = batch.to(device, non_blocking=True)

            # Forward press
            E, phi, Hphi = model(batch)

            # Calculates the loss
            l = loss(E, phi, Hphi)
            running_loss += l.item()

            # Bacward press
            l.backward()
            optimizer.step()

            # Prints current progress
            if(printprogress):
                print(
                    f"\rTraining network: Epoch {j+1}/{epochs}; "
                    f"Batch {i+1}/{num_batches}; "
                    f"Total progress: {100*(j*num_batches + i + 1)/(num_batches*epochs):.1f}%",
                    end="",
                    flush=True,
                )

        # Appends the average running loss at the end of the running_losses list
        else:
            train_losses.append(running_loss/num_batches)

    return model, train_losses

