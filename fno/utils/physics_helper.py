import torch.nn.functional as F
import torch

def second_derivative(f, dt):
    """Returns the second derivative of a batch of functions"""

    # f: (batch_size, N)

    kernel = torch.tensor([1., -2., 1.],
                          device=f.device,
                          dtype=f.dtype).view(1, 1, 3) / dt**2

    # Does the derivative. Cuda compatible
    d2 = F.conv1d(f.unsqueeze(1), kernel).squeeze(1)

    out = torch.empty_like(f)

    # Inner points
    out[:, 1:-1] = d2

    # Boundaries
    out[:, 0] = (2*f[:,0] - 5*f[:,1] + 4*f[:,2] - f[:,3]) / dt**2
    out[:, -1] = (2*f[:,-1] - 5*f[:,-2] + 4*f[:,-3] - f[:,-4]) / dt**2

    return out

def hamiltonian(f, V, dt):
    """Apply the dimensionless Hamiltonian to a batch of functions."""
    return V*f - second_derivative(f, dt) 

def _dirichlet_hamiltonian(V, dt, n):
    """Build the finite-difference Hamiltonian on the interior grid points."""

    if V.ndim != 2 or V.shape[1] < 3:
        raise ValueError("V must have shape (batch_size, N) with N >= 3.")

    # The two endpoints are fixed to zero, so only N - 2 values are unknown.
    interior = V[:, 1:-1]
    size = interior.shape[1]
    if not 0 <= n < size:
        raise ValueError(f"Energy level n must be between 0 and {size - 1}.")

    inv_dt2 = 1 / dt**2
    H = torch.diag_embed(2 * inv_dt2 + interior)
    idx = torch.arange(size - 1, device=V.device)
    H[:, idx, idx + 1] = -inv_dt2
    H[:, idx + 1, idx] = -inv_dt2
    
    return H


def solve_schrodinger(V, dt, n):
    """
    Solve (-d²/dx² + V) phi = E phi with zero endpoint values.

    V has shape (batch_size, N) and includes both endpoints. The grid spacing
    is dt and n is the zero-based eigenstate index. Returns energy with shape
    (batch_size, 1) and the normalized wavefunction with shape (batch_size, N).
    """
    H = _dirichlet_hamiltonian(V, dt, n)
    energies, states = torch.linalg.eigh(H)
    E = energies[:, n:n+1]

    # The eigensolver uses a Euclidean norm of one. Convert to the grid norm
    # and restore the two fixed endpoints.
    phi = F.pad(states[:, :, n] / dt**0.5, (1, 1))
    return E, phi


def solve_energy(V, dt, n):
    """
    Return the n-th Dirichlet eigenenergy for each potential in V.

    The potential grid includes both endpoints and has spacing dt. The
    returned tensor has shape (batch_size, 1), and n=0 is the ground state.
    """
    H = _dirichlet_hamiltonian(V, dt, n)
    return torch.linalg.eigvalsh(H)[:, n:n+1]
