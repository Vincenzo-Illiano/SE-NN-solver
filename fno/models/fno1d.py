import torch
import torch.nn as nn
from ..utils import helper

# Class defining the 1d fourier operator. 
# This requires a integer number of modes (number of frequencies on which the
# network is going to decompose the input), N (number of points of the input),
# A (size of the input set, by default A = 1, meaning the set will be [-1, 1]) and
# hidden (list of integers defining the size of the hidden layers).
# When running forward(), the model will return E (energy), phi (the wavefunction),
# and Hphi (the hamiltonian applied to the wavefunction), as these may be all used in
# training losses
class fno1d(nn.Module):
    def __init__(self, modes, N, hidden, A=1, name="fno1d"):
        super().__init__()

        # Builds the class
        self.modes = modes
        self.N = N
        self.dt = 2*A / ( N - 1)
        self.hidden = hidden

        # Builds the network from the config file
        sizes = (
            [2*modes]
            + hidden
            + [2*modes]
        )
        layers = []
        for in_features, out_features in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())

        # Removes the last activation function
        layers.pop()

        self.net = nn.Sequential(*layers)

        # Mask needed to impose phi = 0 at boundaries.
        # Using a discontinuous function in this case
        mask = torch.ones(self.N)
        mask[0]=0
        mask[-1]=0
        self.register_buffer("boundary_mask", mask)

    # Forward function of the network
    def forward(self, x):

        # Real FFT of input
        F = torch.fft.rfft(x)

        # takes the first modes
        low = F[:, :self.modes]

        # Divides complex and real values
        low_real = torch.view_as_real(low)      # (batch,modes,2)
        low_real = low_real.flatten(1)          # (batch,2modes)

        # Neural network
        out = self.net(low_real)

        # Goes back to complex mode
        out = out.view(-1, self.modes, 2)      # (batch, modes, 2)
        out = torch.view_as_complex(out)

        # Same shape as F needed to do inverse FFT
        F_new = torch.zeros_like(F)
        F_new[:, :self.modes] = out

        # inverse FFT
        phi = torch.fft.irfft(F_new, n=self.N)

        # Imposes phi = 0 at boundaries using the mask defined earlier
        phi = phi * self.boundary_mask

        # Normalizes the output function
        integral = torch.sqrt(self.dt * torch.sum(phi**2, dim=1, keepdim=True))
        phi = phi / integral

        # Calculates E based on phi using E = <phi|H|phi>
        Hphi = helper.hamiltonian(phi, x, self.dt)
        E = self.dt * torch.sum(phi * Hphi, dim=1, keepdim=True)

        return E, phi, Hphi

    # Saves the model parameters and information to PATH
    # in particular, saves:
    # 'model_state_dict' state dict containing the model parameters
    # 'modes' number of modes of the model
    # 'N' number of points that the model can take as input function
    # 'hidden' list describing the amount of hidden layer parameters
    # 'dt' discretization step used by this model on the set [-A, A]
    def saveModel(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'modes': self.modes,
            'N': self.N,
            'hidden': self.hidden,
            'dt': self.dt
            }, path)