import math

import torch


def testing_loss(m_E, m_phi, E, phi, a=1):
    """Return the weighted energy MSE plus the sign-aligned wavefunction MSE."""

    # A real eigenfunction and its negative describe the same state.
    overlap = torch.sum(m_phi * phi, dim=-1, keepdim=True)
    aligned_phi = torch.where(overlap < 0, -phi, phi)

    return a * torch.mean((m_E - E)**2) + torch.mean((m_phi - aligned_phi)**2)


@torch.no_grad()
def testFno1d(model, testingset, a=1, device="cuda", printprogress=False):
    """
    Test a model and return a list of scalar losses on the CPU, one per batch.

    Arguments:
    model -- model to test
    testingset -- labeled dataset produced by generate_testing_set
    a -- non-negative weight of the energy MSE
    device -- device used for testing (default cuda)
    printprogress -- whether to print the testing progress
    """

    # Does a lot of various checks to make sure that the provided data is in the right format
    required_keys = {"function", "phi", "E"}
    if not isinstance(testingset, dict) or not required_keys.issubset(testingset):
        raise ValueError("Expected a labeled testing set with potentials, wavefunctions, and energies.")

    func_set = testingset["function"]
    phi_set = testingset["phi"]
    E_set = testingset["E"]
    if not all(torch.is_tensor(tensor) for tensor in (func_set, phi_set, E_set)):
        raise ValueError("The potentials, wavefunctions, and energies must be tensors.")
    if func_set.ndim != 3:
        raise ValueError("The testing potentials must have shape (num_batches, batch_size, N).")

    num_batches, batch_size, N = func_set.shape
    for key, size in zip(("num_batches", "batch_size", "N"), func_set.shape):
        if key in testingset and testingset[key] != size:
            raise ValueError(f"The testing set metadata for {key} does not match its tensors.")

    if num_batches < 1 or batch_size < 1:
        raise ValueError("The testing set must contain at least one sample.")
    if N != model.N:
        raise ValueError(f"Model expects N={model.N}, but the testing set has N={N}.")
    # Older datasets may omit A. For those, the caller supplies a matching model.
    if "A" in testingset and not math.isclose(testingset["A"], model.A):
        raise ValueError(f"Model uses A={model.A}, but the testing set uses A={testingset['A']}.")
    if phi_set.shape != func_set.shape or E_set.shape != (num_batches, batch_size, 1):
        raise ValueError("Testing tensors do not match the dataset's batch and grid sizes.")

    # Loads the model to the device and sets it in evaluation mode (testing)
    model.to(device)
    model.eval()

    losses = []
    for i, (potentials, phi, E) in enumerate(zip(func_set, phi_set, E_set), start=1):

        # Loads all the batch data
        potentials = potentials.to(device, non_blocking=True)
        phi = phi.to(device, non_blocking=True)
        E = E.to(device, non_blocking=True)

        # Calculates the model output and the loss
        m_E, m_phi, _ = model(potentials)
        losses.append(testing_loss(m_E, m_phi, E, phi, a).cpu())

        # Prints progress
        if printprogress:
            print(f"\rTesting network: {i}/{num_batches} ({100*i/num_batches:.1f}%)",
                  end="", flush=True)

    if printprogress:
        print()

    return losses