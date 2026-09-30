import torch
import torch.nn as nn
from ..utils import physics_helper

class fno1d(nn.Module):
    """
    Class defining the 1d fourier operator. 

    Attributes:
    modes -- number of frequencies on which the network is going to decompose the input
    N -- number of points of the input functions
    hidden -- list of integers defining the size of the hidden layers
    dt -- discretization step of the set on which the shrodinger eq. is solved
    name -- name of the model

    Methods:
    forward(x) -- given an input batch of functions x, returns energies, wavefunctions, and 
        the wavefunctions hamiltonians (adimensional)
    
    save(path) -- Saves the model parameters and information to PATH
    """
    def __init__(self, modes, N, hidden, A=1, name="fno1d"):
        """Constructs the fno1d class"""

        super().__init__()

        # Builds the class
        self.modes = modes
        self.N = N
        self.dt = 2*A / ( N - 1)
        self.hidden = hidden
        self.name = name

        # Checks that the parameters have correct values
        if not isinstance(modes, int) or modes <= 0:
            raise ValueError("modes needs to be an integer greater than 0")
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N needs to be an integer greater than 0")
        for n in hidden:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("Every element of hidden needs to be an integer greater than 0")
        if not isinstance(A, float) or A <= 0:
            raise ValueError("A needs to be a float greater than 0")

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

    def forward(self, x):
        """
        Forward function of the network.
        x should be a batch of functions like (batch_size, n), where n is the number of points of the functions.

        Returns the energy, wavefuntion and wavefunction's hamiltonian (adimensional)
        """

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
        Hphi = physics_helper.hamiltonian(phi, x, self.dt)
        E = self.dt * torch.sum(phi * Hphi, dim=1, keepdim=True)

        return E, phi, Hphi

    def saveModel(self, path):
        """
        Saves the model parameters and information to PATH
        in particular, saves:
        'model_state_dict' state dict containing the model parameters
        'modes' number of modes of the model
        'N' number of points that the model can take as input function
        'hidden' list describing the amount of hidden layer parameters
        'dt' discretization step used by this model on the set [-A, A]
        
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'modes': self.modes,
            'N': self.N,
            'hidden': self.hidden,
            'dt': self.dt
            }, path)

    @staticmethod
    def set_create_arguments(parser):
        """
        Static function needed in order to be able to create a fno1d using scripts/create_model.py.
        Given a parser, this will add the necessary arguments in order for the network to be created with
        inline console arguments. 
        """
        parser.add_argument(
            "-n", "--name",
            help="Name of the model",
            type=str,
            default="fno1d"
        )

        parser.add_argument(
            "-N", "--N",
            help="Number of points in a function that the model takes as an input",
            type=int,
            required=True
        )

        parser.add_argument(
            "-M", "--modes",
            help="Number of frequency modes that the model breaks the input into",
            type=int,
            required=True
        )

        parser.add_argument(
            "--hidden",
            help="Number of hidden layers parameters. For example -h 10 20 10 would create a model with 3 hidden layers" \
            "with respectively 10, 20 and 10 parameters",
            nargs="+",
            type=int,
            required=True
        )

        parser.add_argument(
            "-A", "--inputwidth",
            help="The functions that the model takes as an input should be interpreted as functions on the interval (-A, A)." \
            "By default A=1, so the input functions are defined on the set (-1,1)",
            type=float,
            default=1
        )

    @classmethod
    def from_args(cls, args):
        """
        Given parser arguments defined in set_create_parsing, creates an instance of the class from these.
        """

        # returns a fno1d with the arguments specified by the parser arguments
        return cls(
            N=args.N,
            modes=args.modes,
            hidden = args.hidden,
            A=float(args.inputwidth),
            name=args.name
        )


