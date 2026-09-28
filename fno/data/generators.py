import torch.nn.functional as F
import torch
from ..utils import physics_helper

# Contains functions used to generate generic datasets, testing and training sets.

# 
# n is the number of points of the function (discretization size).
def random_smooth(batch_size, n, sigma=8, device="cuda"):
    """
    Generates a batch of random torch tensors representing continous functions.

    keyword arguments:
    batch_size -- number of functions inside a batch
    n -- number of points inside a single functions
    sigma -- smoothing radious (default 8)
    device -- device used to generate the batch. Could be "cuda" or "cpu" (default cuda)
    """

    # Generates white noise
    x = torch.randn(batch_size, 1, n, device=device)

    # Smooths out the white noise using a moving average
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1

    t = torch.arange(k, device=device) - k // 2

    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    y = F.conv1d(
        F.pad(x, (k // 2, k // 2), mode="reflect"),
        kernel.view(1, 1, -1)
    ).squeeze(1)

    # Window equal to zero at the boundaries
    window = torch.hann_window(n, periodic=False, device=device)
    y *= window

    # normalizes the functions
    y /= y.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)

    return y

# 
def random_polynomials(M, N, degree=8, device="cuda"):
    """
    Generates a batch of M random polynomials of degree with N points.

    keyword arguments:
    M -- number of functions inside a batch
    N -- number of points inside a single functions
    degree -- degree of used polynomials
    device -- device used to generate the batch. Could be "cuda" or "cpu" (default cuda)
    """
    
    x = torch.linspace(-1, 1, N, device=device)

    # (M, degree+1)
    coeffs = torch.randn(M, degree + 1, device=device)

    # (degree+1, N)
    powers = torch.stack([x**k for k in range(degree + 1)])

    # (M, N)
    f = coeffs @ powers

    # Makes sure that f = 0 at boundaries
    mask = 1 - x**8
    f *= mask

    # normalizes the functions
    f /= f.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)

    return f

# 
def random_gaussian_wells(M, N, max_wells=4, well_steep = 2, device="cuda"):
    """
    Generates a batch of randomly placed and deep gausian wells.

    keyword arguments:
    M -- number of functions inside a batch
    N -- number of points inside a single functions
    max_wells -- maximum number of wells in a function (default 4)
    well_steep -- How steep is a well (default 2)
    device -- device used to generate the batch. Could be "cuda" or "cpu" (default cuda)
    """

    # Assumes the function is between -1 and 1 (can be stretched)
    x = torch.linspace(-1, 1, N, device=device)
    x = x[None, None, :]              # (1,1,N)

    K = max_wells
    well_steep = 2*well_steep # well steep has to be even

    # Choses the positions of the centers of the well between -1 and 1
    centers = torch.rand(M, K, device=device) * 2 - 1

    # Generates random widths between 0.05 and 0.30
    widths = 0.05 + 0.25 * torch.rand(M, K, device=device)
    # Generates random depths between -1 and +1
    depths = 2 * torch.rand(M, K, device=device) - 1

    centers = centers[:, :, None]
    widths = widths[:, :, None]
    depths = depths[:, :, None]

    # Creates the wells
    wells = depths * torch.exp(
        -(x - centers)**well_steep / (2 * widths**well_steep)
    )

    # Sums them
    V = wells.sum(dim=1)

    # Makes sure that the function is 0 at boundaries
    mask = 1 - torch.linspace(-1,1,N,device=device)**8
    V *= mask

    return V

# Generates and saves a dataset
def generate_set( num_batches, batch_size, N, device="cuda", type="mixed", printprogess=False):
    """
    Generates and returns a function dataset.

    keyword arguments:
    num_batches -- number of batches in the dataset
    batch_size -- number of functions saved in one batch
    N -- number of points in one function
    device -- device used to generate the batch. Could be "cuda" or "cpu" (default cuda)
    type -- dataset type. Can be "mixed", "poly", "well" or "smooth" (default mixed)
    printprogress -- whether to print the progress of the generation (default False)
    """

    # Generates an empy dataset
    dataset = torch.empty(
        num_batches,
        batch_size,
        N,
        dtype=torch.float32
    )

    for b in range(num_batches):

        # Generates three batches of different type
        poly   = random_polynomials(batch_size, N, device=device)
        smooth = random_smooth(batch_size, N, device=device, sigma=32)
        wells  = random_gaussian_wells(batch_size, N, device=device, max_wells=2)

        # Prints current progress
        if(printprogess):
            print(f"\rGenerating dataset: {b+1}/{num_batches} ({100*b/num_batches:.1f}%)", end="", flush=True)

        # Creates the dataset for non mixed types
        if(type == "well"):
            dataset[b] = wells.cpu()
            continue
        elif(type == "smooth"):
            dataset[b] = smooth.cpu()
            continue
        elif(type == "poly"):
            dataset[b] = poly.cpu()
            continue
        elif(type != "mixed"):
            raise ValueError("type %s is not an available dataset type" % type)

        # Creates the dataset for mixed types
        # Generates a batch three random coefficients between 0 and 1
        weights = torch.rand(batch_size, 3, device=device)

        # Decides if a generator is used
        active = (torch.rand(batch_size, 3, device=device) < 0.7).float()

        # Avoids all being 0
        inactive = active.sum(dim=1) == 0 # boolean mask to find inactive spots
        active[inactive, torch.randint(0, 3, (inactive.sum(),), device=device)] = 1

        # Creates the batch
        weights *= active
        batch = (
            weights[:, 0:1] * poly +
            weights[:, 1:2] * smooth +
            weights[:, 2:3] * wells
        )

        # Updates the dataset with the current batch
        dataset[b] = batch.cpu()

    return dataset

def generate_training_set( filename, num_batches, batch_size, N, device="cuda", type="mixed", printprogress=False):
    """
    Generates and saves a dataset of random functions for training. 
    The result will be a dictionare containing:
    'function': the functions dataset
    'batch_size': batch_size,
    'N': N,
    'type': type

    keyword arguments:
    filename -- path of the output file
    num_batches -- number of batches in the dataset
    batch_size -- number of functions saved in one batch
    N -- number of points in one function
    device -- device used to generate the batch. Could be "cuda" or "cpu" (default cuda)
    type -- dataset type. Can be "mixed", "poly", "well" or "smooth" (default mixed)
    printprogress -- whether to print the progress of the generation (default False)
    """

    # Generates the set
    dataset = generate_set(num_batches, batch_size, N, device=device, type=type, printprogress=printprogress)

    # Saves the set
    torch.save({"function":dataset,
                "num_batches": num_batches,
                "batch_size": batch_size,
                "N": N,
                "type": type}, filename)

def generate_testing_set( filename, num_batches, batch_size, N, A=1, device="cuda", type="mixed", printprogress=False):
    """
    Generates and saves a dataset of random functions for testing. 
    The result will be a dictionare containing:
    'function': the dataset containing the random functions
    'phi' a dataset containing the wavefunction corresponding to each random function
    'E' corresponding ground energy of each wavefunction 
    'batch_size': batch_size,
    'N': N,
    'type': type

    keyword arguments:
    filename -- path of the output file
    num_batches -- number of batches in the dataset
    batch_size -- number of functions saved in one batch
    N -- number of points in one function
    A -- size of the set where the schrodinger function will be solved (default [-1, 1])
    device -- device used to generate the batch. Could be "cuda" or "cpu" (default cuda)
    type -- dataset type. Can be "mixed", "poly", "well" or "smooth" (default mixed)
    printprogress -- whether to print the progress of the generation (default False)
    """

    # Generates the functions set
    function_set = generate_set( num_batches, batch_size, N, device=device, type=type, printprogess=printprogress)

    # Creates the solved sets
    energy_set = torch.empty(
        num_batches,
        batch_size,
        1,
        dtype=torch.float32
    )
    solved_set = torch.empty(
        num_batches,
        batch_size,
        N,
        dtype=torch.float32
    )

    # For each function in the set solves the shrodinger equation
    for i, batch in enumerate(function_set):

        # Moves the batch to the used device
        batch = batch.to(device, non_blocking=True)

        # Founds the n-th energy level for the current batch
        with torch.no_grad():
            E, phi = physics_helper.solve_schrodinger(batch, 2*A/(N-1), 0)

        # Loads 
        energy_set[i] = E.cpu()
        solved_set[i] = phi.cpu()

        # Prints current progress
        if(printprogress):
            print(f"\rGenerating testing set: {i+1}/{num_batches} ({100*i/num_batches:.1f}%)", end="", flush=True)

    # Saves everything in a dictionary
    torch.save({
        "function": function_set,
        "phi": solved_set,
        "E": energy_set,
        "type": type,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "N": N,
        "A": A
    }, filename)

def generate_energy_set( input_filename, output_filename, A=1, device = "cuda", n = 0, printprogress=False):
    """
    Generates a dataset containing the n-th energies for the potentials stored in the input_filename dataset.
    The result will be a tensor of shape (num_batches, batch_size, 1).

    keyword arguments:
    input_filename -- path of the input dataset 
    output_filename -- path of the output dataset
    A -- size of the set where the schrodinger function will be solved (default [-1, 1])
    device -- device used to generate the batch. Could be "cuda" or "cpu" (default cuda)
    n -- which energy level to compute (default 0)
    printprogress -- whether to print the progress of the generation (default False)
    """
    # Loads the training dataset
    data = torch.load( input_filename, map_location="cpu")

    # Imports info from the set
    trainingset = data['function']
    num_batches = data['num_batches']
    batch_size = data['batch_size']
    N = data['N']

    # Generates an empty dataset
    dataset = torch.empty(
        num_batches,
        batch_size,
        1,
        dtype=torch.float32
    )

    # For each batch in the training set, finds the energy of the n-th state
    for i, batch in enumerate(trainingset):

        # Moves the batch to the used device
        batch = batch.to(device, non_blocking=True)

        # Founds the n-th energy level for the current batch
        with torch.no_grad():
            E = physics_helper.solve_energy(batch, 2*A/(N-1), n)

        # Loads 
        dataset[i] = E.cpu()

        # Prints current progress
        if(printprogress):
            print(f"\rGenerating dataset: {i+1}/{num_batches} ({100*i/num_batches:.1f}%)", end="", flush=True)

    # Saves the dataset
    torch.save(dataset, output_filename)

def shuffleDataset(dataset, num_batches, batch_size, N):
    """Shuffles the functions inside a dataset"""
    
    # Shuffles
    dataset = dataset.reshape(-1, N)
    dataset = dataset[torch.randperm(dataset.size(0))]
    dataset = dataset.reshape(num_batches, batch_size, N)