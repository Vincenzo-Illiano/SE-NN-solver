import torch

def minimumE_loss(E):
    """Loss function that minimizes the energy (minimum energy loss)"""

    return torch.mean(E)

def minimumErr_loss(E, phi, Hphi):
    """Loss function that minimizes (Hphi - Ephi)^2 (minimum error loss)"""

    return torch.mean((Hphi-E*phi)**2)
