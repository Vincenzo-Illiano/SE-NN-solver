import matplotlib.pyplot as plt
import numpy as np
import torch

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