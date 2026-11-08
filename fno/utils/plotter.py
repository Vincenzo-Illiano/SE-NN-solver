import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models.fno1d import fno1d

def view_dataset(input_filename, index, A=1):
    """ plots the function at position (num_batch, num_fun) in a dataset"""

    # Loads the training dataset
    data = torch.load( input_filename, map_location="cpu")

    trainingset = data['function']
    num_batches = data['num_batches']
    batch_size = data['batch_size']
    N = data['N']

    if(index[0] > num_batches or index[1] > batch_size):
        raise ValueError("index out of range")

    f = trainingset[index[0], index[1], :].detach().cpu().numpy()
    t = np.linspace(-A, A, N)

    plt.plot(t, f)
    plt.show()


def plot_model_solution(model_filename, testing_filename, index=(0, 0), *, legacy_A=None):
    """
    Plot the prediction and reference ground state for one testing potential.

    Arguments:
    model_filename -- path to the saved fno1d checkpoint
    testing_filename -- path to the labeled testing dataset
    index -- zero-based (batch_index, sample_index), default (0, 0)
    legacy_A -- interval half-width for an older checkpoint saved as a bare state dict

    Shows the potential, both wavefunctions, and their energies. Returns the
    Matplotlib figure so it can be saved or further customized.
    """
    data = torch.load(testing_filename, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not {"function", "phi", "E"}.issubset(data):
        raise ValueError("Expected a labeled testing set with potentials, wavefunctions, and energies.")

    potentials = data["function"]
    num_batches, batch_size, N = potentials.shape
    batch_index, sample_index = index
    if not (0 <= batch_index < num_batches and 0 <= sample_index < batch_size):
        raise ValueError(f"Testing index {index} is out of range for shape {potentials.shape}.")

    model = fno1d.from_file(model_filename, legacy_A=legacy_A)
    if N != model.N:
        raise ValueError(f"Model expects N={model.N}, but the testing set has N={N}.")
    if "A" in data and not math.isclose(data["A"], model.A):
        raise ValueError(f"Model uses A={model.A}, but the testing set uses A={data['A']}.")

    potential = potentials[batch_index, sample_index]
    reference_phi = data["phi"][batch_index, sample_index]
    reference_energy = data["E"][batch_index, sample_index].item()

    # The model expects a batch, even when evaluating a single potential.
    model.eval()
    with torch.inference_mode():
        energy, phi, _ = model(potential.unsqueeze(0))
        predicted_phi = phi[0]

        # Real eigenfunctions are defined up to an overall sign.
        if torch.dot(predicted_phi, reference_phi) < 0:
            predicted_phi = -predicted_phi

    x = np.linspace(-model.A, model.A, N)
    fig, (potential_ax, solution_ax) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    fig.suptitle(f"Testing potential at batch {batch_index}, sample {sample_index}")

    potential_ax.plot(x, potential.numpy(), color="0.3")
    potential_ax.set_ylabel(r"$V(x)$")

    solution_ax.plot(x, reference_phi.numpy(), label=f"Reference (E = {reference_energy:.6g})")
    solution_ax.plot(x, predicted_phi.numpy(), "--", label=f"Model (E = {energy.item():.6g})")
    solution_ax.set_xlabel(r"$x$")
    solution_ax.set_ylabel(r"$\phi(x)$")
    solution_ax.legend()

    fig.tight_layout()
    plt.show()
    return fig
