import torch

def testing_loss(m_E, m_phi, E, phi, a=1):
    """
    This loss function is intended for testing purposes and cannot be used for training.
    This checks how much the values returned by the model are close to the expected values,
    therefore it returns a*(m_E - E)**2 + (m_phi - phi)**2
    """

    return a*torch.mean( (m_E - E)**2 ) + torch.mean( (m_phi - phi)**2 )

def testFno1d(model, testingset, a=1, device="cuda", printprogress=False):
    """
    Tests a Fno1d model on a testingset and returns a list containing the test loss for each batch.
    
    Arguments:
    model -- model to test
    testingset -- testing set 
    a -- The loss function is a*(model_E - correct_E)^2 + (model_phi - correct_phi)^2 (default a=1)
    device -- device used for testing. Can be "cuda" or "cpu". (default cuda)
    printprogress -- whether to print the training progress (default False)
    """

    # Loads the model to the device
    model.to(device)

    # Gets information about the training set
    func_set = testingset['function']
    phi_set = testingset["phi"]
    E_set = testingset["E"]
    num_batches = testingset['num_batches']
    N = testingset['N']

    # Makes sure that the functions that the model takes have the same size as the ones in the dataset
    if(N != model.N):
        raise ValueError(f"Model expects function size {model.N}, but trainingset has function size {N}")

    # Makes sure to be in testing mode
    model.eval()

    # Begins testing
    losses = []
    for i in range(num_batches):

        # Takes the current batch
        f = func_set[i].to(device, non_blocking=True)
        phi = phi_set[i].to(device, non_blocking=True)
        E = E_set[i].to(device, non_blocking=True)

        # Calculates the model phi and E
        m_E, m_phi, m_Hphi = model(f)

        # Appends the testing loss to the losses
        losses.append(testing_loss(m_E, m_phi, E, phi, a))

        # Prints progress
        if(printprogress):
            print(f"\rTesting network: {i+1}/{num_batches} ({100*(i+1)/num_batches:.1f}%)", end="", flush=True)

    if(printprogress):
        print()

    # Returns the losses
    return losses

    