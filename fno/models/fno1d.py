import torch
import torch.nn as nn

# Class defining the 1d fourier
class fno1d(nn.Module):
    def __init__(self, modes, N, A):
        super().__init__()

        # Builds the class
        self.modes = modes
        self.N = N
        self.dt = 2*A / ( N - 1)

        # Builds the network from the config file
        sizes = (
            [2*config.MODES]
            + config.HIDDEN_LAYERS
            + [2*config.MODES]
        )
        layers = []
        for in_features, out_features in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())

        # Removes the last activation function
        layers.pop()

        self.net = nn.Sequential(*layers)

        # Mask needed to impose phi = 0 at boundaries
        mask = torch.ones(self.N)
        mask[0]=0
        mask[-1]=0
        self.register_buffer("boundary_mask", mask)

    def forward(self, x):

        # Real FFT of input
        F = torch.fft.rfft(x)

        # takes the first modes
        low = F[:, :self.modes]

        # Divides comples and real values
        low_real = torch.view_as_real(low)      # (batch,M,2)
        low_real = low_real.flatten(1)          # (batch,2M)

        # Neural network
        out = self.net(low_real)

        # Goes back to comples mode
        out = out.view(-1, self.modes, 2)
        out = torch.view_as_complex(out)

        # Same shape as F needed to do inverse FFT
        F_new = torch.zeros_like(F)
        F_new[:, :self.modes] = out

        # inverse FFT
        phi = torch.fft.irfft(F_new, n=self.N)

        # Imposes phi = 0 at boundaries
        phi = phi * self.boundary_mask

        # Makes sure that the output function is normalized
        integral = torch.sqrt(self.dt * torch.sum(phi**2, dim=1, keepdim=True))
        phi = phi / integral

        # Calculates E based on phi using E = <phi|H|phi>
        Hphi = helper.hamiltonian(phi, x, self.dt)
        E = self.dt * torch.sum(phi * Hphi, dim=1, keepdim=True)

        return E, phi, Hphi