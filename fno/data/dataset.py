import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import network.helper as helper
import network.config as config

# Generates a batch of random torch tensors representing continous functions.
# n is the number of points of the function (discretization size).
def random_smooth(batch_size, n, sigma=8, device="cuda"):

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

# Generates a batch of M random polynomials of degree with N points using cuda
def random_polynomials(M, N, degree=8, device="cuda"):
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

# Generates a batch of randomly placed and deep gausian wells using cuda
def random_gaussian_wells( M, N, max_wells=4, well_steep = 2, device="cuda"):

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
def generate_set( num_batches, batch_size, N, device="cuda", type="mixed"):

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

# Generates and saves a dataset of functions for training. This is a tensor of
# shape (num_batches, batch_size, N)
# Types of datasets can be: "mixed", "smooth", "well", "poly"
def generate_training_set( filename, device="cuda", type="mixed"):

    print('Generating %s type dataset' % type)

    # Imports data from config file
    NUM_BATCHES = config.NUM_BATCHES
    BATCH_SIZE = config.BATCH_SIZE
    N = config.N

    # Generates the set
    dataset = generate_set(NUM_BATCHES, BATCH_SIZE, N, type=type)

    # Saves the set
    torch.save(dataset, filename)
    print(f"\nDataset saved in '{filename}'")

# Generates and saves a dataset of functions with respective energies and solved
# shrodinger equation in a tensor of shape. This will save a dictionary containing
# the three tensors
def generate_testing_set( filename, device="cuda", type="mixed"):

    print('Generating %s type testing set' % type)

    # Imports data from config file
    TESTING_NUM_BATCHES = config.TESTING_NUM_BATCHES
    TESTING_BATCH_SIZE = config.TESTING_BATCH_SIZE
    A = config.A
    N = config.N

    # Generates the functions set
    print("Generating functions set...")
    function_set = generate_set( TESTING_NUM_BATCHES, TESTING_BATCH_SIZE, N, device=device, type=type)

    # Creates the solved sets
    energy_set = torch.empty(
        TESTING_NUM_BATCHES,
        TESTING_BATCH_SIZE,
        1,
        dtype=torch.float32
    )
    solved_set = torch.empty(
        TESTING_NUM_BATCHES,
        TESTING_BATCH_SIZE,
        N,
        dtype=torch.float32
    )

    # For each function in the set solves the shrodinger equation
    print("Generating testing set...")
    for i, batch in enumerate(function_set):

        # Moves the batch to the used device
        batch = batch.to(device, non_blocking=True)

        # Founds the n-th energy level for the current batch
        with torch.no_grad():
            E, phi = helper.solve_schrodinger(batch, 2*A/(N-1), 0)

        # Loads 
        energy_set[i] = E.cpu()
        solved_set[i] = phi.cpu()

        # Prints current progress
        print(f"\rGenerating testing set: {i+1}/{TESTING_NUM_BATCHES} ({100*i/TESTING_NUM_BATCHES:.1f}%)", end="", flush=True)

    # Saves everything in a dictionary
    torch.save({
        "function": function_set,
        "phi": solved_set,
        "E": energy_set,
        "type": type
    }, filename)
    print("Testing set saved correctly")

# Generates a dataset containing the n-th energies for the potentials
# stored in input_filename
def generate_energy_set( input_filename, output_filename, device = "cuda", n = 0):

    # Loads the training dataset
    print("Loading training set...")
    trainingset = torch.load( input_filename, map_location="cpu")
    print("Training set loaded correctly")

    # Imports data from config file
    NUM_BATCHES = config.NUM_BATCHES
    BATCH_SIZE = config.BATCH_SIZE
    N = config.N
    A = config.A

    # Generates an empty dataset
    dataset = torch.empty(
        NUM_BATCHES,
        BATCH_SIZE,
        1,
        dtype=torch.float32
    )

    # For each batch in the training set, finds the energy of the n-th state
    for i, batch in enumerate(trainingset):

        # Moves the batch to the used device
        batch = batch.to(device, non_blocking=True)

        # Founds the n-th energy level for the current batch
        with torch.no_grad():
            E = helper.solve_energy(batch, 2*A/(N-1), n)

        # Loads 
        dataset[i] = E.cpu()

        # Prints current progress
        print(f"\rGenerating dataset: {i+1}/{NUM_BATCHES} ({100*i/NUM_BATCHES:.1f}%)", end="", flush=True)

    # Saves the dataset
    torch.save(dataset, output_filename)
    print(f"\nDataset saved in '{output_filename}'")

# plots the function at position (num_batch, num_fun) in a dataset
def view_dataset(input_filename, index):

    # Imports data from config file
    NUM_BATCHES = config.NUM_BATCHES
    BATCH_SIZE = config.BATCH_SIZE
    N = config.N
    A = config.A

    # Loads the training dataset
    print("Loading training set...")
    trainingset = torch.load( input_filename, map_location="cpu")
    print("Training set loaded correctly")

    if(index[0] > NUM_BATCHES or index[1] > BATCH_SIZE):
        raise ValueError("index out of range")

    f = trainingset[index[0], index[1], :].detach().cpu().numpy()
    t = np.linspace(-A, A, N)

    plt.plot(t, f)
    plt.show()

# Shuffles all the functions inside a dataaset
def shuffleDataset(dataset):

    # Imports data from config file
    NUM_BATCHES = config.NUM_BATCHES
    BATCH_SIZE = config.BATCH_SIZE
    N = config.N

    # Shuffles
    dataset = dataset.reshape(-1, N)
    dataset = dataset[torch.randperm(dataset.size(0))]
    dataset = dataset.reshape(NUM_BATCHES, BATCH_SIZE, N)