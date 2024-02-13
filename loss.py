import torch
from torch.distributions import kl_divergence, Normal
from torch.nn import NLLLoss
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


def loss_function_vae(
    dec: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    stdev: torch.Tensor,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """
    Calculate the loss function for a Variational Autoencoder (VAE).

    Args:
        dec (torch.Tensor): Decoded output tensor.
        x (torch.Tensor): Input tensor.
        mu (torch.Tensor): Mean values from the encoder.
        stdev (torch.Tensor): Standard deviation values from the encoder.
        kl_weight (float, optional): Weight for the KL divergence term. Defaults to 1.0.

    Returns:
        torch.Tensor: Total loss value.

    """
    # Define the prior distribution
    mean = torch.zeros_like(
        mu
    )  # tensor with the same shape as mu but filled with zeros
    scale = torch.ones_like(
        stdev
    )  # tensor with the same shape as stdev but filled with ones

    # Calculate KL divergence
    KLD = kl_divergence(Normal(mu, stdev), Normal(mean, scale)).mean(dim=1)

    # Reconstruction loss
    reconst_loss = F.mse_loss(dec, x, reduction="none").mean(dim=1)

    # Total loss
    total_loss = (reconst_loss + kl_weight * KLD).sum(dim=0)

    return total_loss


'''
def loss_function_classification(pred, target, class_weights):
    """
    Compute negative log likelihood loss.

    :param pred: predictions
    :param target: actual classes
    :param class_weights: weights - one per class
    :return: Weighted prediction loss. Summed for all the samples, not averaged.
    """
    loss_function = NLLLoss(weight=class_weights, reduction="none")
    return loss_function(pred, torch.argmax(target, dim=1)).sum(dim=0)
'''

"""
def gaussian_kernel(a, b, kernel_length=1.0, device="cpu"):
    x = a.view(len(a), -1)
    y = b.view(len(b), -1)
    dim = x.size(1)
    x = x.unsqueeze(1)  # shape: (len(a), 1, dim)
    y = y.unsqueeze(0)  # shape: (1, len(b), dim)
    kernel = torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * kernel_length**2))
    return kernel


def compute_mmd_loss(z, device="cpu"):
    true_samples = torch.randn(z.size(0), z.size(1), device=device)
    zz_kernel = gaussian_kernel(z, z, device=device)
    tt_kernel = gaussian_kernel(true_samples, true_samples, device=device)
    zt_kernel = gaussian_kernel(z, true_samples, device=device)
    mmd = zz_kernel.mean() + tt_kernel.mean() - 2 * zt_kernel.mean()
    return mmd
"""

# Francesco's version


def loss_function_classification(
    pred: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor
) -> torch.Tensor:
    """
    Compute cross-entropy loss.

    Args:
        pred (torch.Tensor): Predictions (logits before softmax).
        target (torch.Tensor): Actual classes (as indices, not one-hot encoded).
        class_weights (torch.Tensor): Weights - one per class.

    Returns:
        torch.Tensor: Weighted prediction loss. Summed for all the samples, not averaged.

    """
    loss_function = CrossEntropyLoss(weight=class_weights, reduction="none")

    # Assuming target is one-hot encoded, converting to class indices
    target_indices = torch.argmax(target, dim=1)

    return loss_function(pred, target_indices).sum()
