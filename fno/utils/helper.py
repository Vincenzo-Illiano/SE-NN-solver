import torch.nn.functional as F
import torch

# Does the second derivative of a batch of functions 
def second_derivative(f, dt):
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

# Applies the hamiltonian operator to a batch of functions
# Note that this is whithout constant, so the whole thing is adimensional
def hamiltonian(f, V, dt):
    return V*f - second_derivative(f, dt) 

def solve_schrodinger(V, dt, n):
    """
    Solves

        (-d²/dx² + V) phi = E phi

    using finite differences.

    Parameters
    ----------
    V : torch.Tensor
        Shape (M, N) representing batch of potentials.

    dt : float
        Spatial step.

    n : integer
        Energy level

    Returns
        E   : (M, 1)
        phi : (M, N)
    """

    device = V.device
    dtype = V.dtype

    inv_dt2 = 1.0 / (dt * dt)

    # Checks that V has shape (M, N)
    if V.ndim == 2:
        M, N = V.shape

        # Creates the batched hamiltonian matrix
        diag = 2.0 * inv_dt2 + V
        off = -inv_dt2 * torch.ones(N - 1, device=device, dtype=dtype)

        H = torch.diag_embed(diag)

        idx = torch.arange(N - 1, device=device)
        H[:, idx, idx + 1] = off
        H[:, idx + 1, idx] = off

        # Diagonalization
        E, phi = torch.linalg.eigh(H)

        # Returns the n-th state
        E = E[:, n:n+1]
        phi = phi[:, :, n]

        # Normalizes the result
        norm = torch.sqrt(torch.sum(phi**2, dim=1, keepdim=True) * dt)
        phi = phi / norm

        return E, phi

    else:
        raise ValueError("V must have shape (N,) or (M, N)")

def solve_energy(V, dt, n):
    """
    Computes only the n-th energy eigenvalue of

        (-d²/dx² + V) phi = E phi

    using finite differences.

    Parameters
    ----------
    V : torch.Tensor
        Shape (M, N), batch of potentials.

    dt : float
        Spatial step.

    n : int
        Energy level (0 = ground state).

    Returns
    -------
    E : torch.Tensor
        Shape (M, 1), n-th eigenvalue for each potential.
    """

    if V.ndim != 2:
        raise ValueError("V must have shape (M, N)")

    device = V.device
    dtype = V.dtype

    M, N = V.shape

    inv_dt2 = 1.0 / (dt * dt)

    # Diagonal
    diag = 2.0 * inv_dt2 + V

    # Build Hamiltonian batch
    H = torch.diag_embed(diag)

    off = -inv_dt2 * torch.ones(N - 1, device=device, dtype=dtype)

    idx = torch.arange(N - 1, device=device)
    H[:, idx, idx + 1] = off
    H[:, idx + 1, idx] = off

    # Only eigenvalues (faster than eigh)
    E = torch.linalg.eigvalsh(H)

    # n-th eigenvalue
    return E[:, n:n+1]