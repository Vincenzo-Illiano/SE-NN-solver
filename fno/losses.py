import torch

def minimumE_loss(E, phi, Hphi):
    """Loss function that minimizes the energy (minimum energy loss)"""

    return torch.mean(E)

def minimumErr_loss(E, phi, Hphi):
    """Minimize the squared eigenvalue residual on the interior grid points."""

    # The endpoints already satisfy the imposed Dirichlet condition.
    return torch.mean((Hphi[..., 1:-1] - E * phi[..., 1:-1])**2)
