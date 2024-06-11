import torch
from torch.distributions import kl_divergence, Normal
from torch.nn import NLLLoss
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.autograd import Variable


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    x_tensor = (
        x if isinstance(x, torch.Tensor) else x[0]
    )  # Extract tensor if x is a tuple
    y_tensor = (
        y if isinstance(y, torch.Tensor) else y[0]
    )  # Extract tensor if y is a tuple
    x_kernel = compute_kernel(x_tensor, x_tensor)
    y_kernel = compute_kernel(y_tensor, y_tensor)
    xy_kernel = compute_kernel(x_tensor, y_tensor)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def loss_function_vae(
    enc, dec, x, mu, stdev, device, discrepancy_weight=1e-5, discrepancy="KLD"
):
    # sum over genes, mean over samples, like trvae

    mean = torch.zeros_like(
        mu
    )  # tensor with the same shape as mu but filled with zeros
    scale = torch.ones_like(
        stdev
    )  #  tensor with the same shape as stdev but filled with ones
    if discrepancy == "KLD":
        discrepancy_loss = kl_divergence(Normal(mu, stdev), Normal(mean, scale)).mean(
            dim=1
        )  # Regularisation term

    elif discrepancy == "MMD":
        true_samples = Variable(
            torch.randn(enc.shape[0], enc.shape[1]), requires_grad=False
        )
        discrepancy_loss = compute_mmd(true_samples.to(device), enc.to(device))

    reconst_loss = F.mse_loss(dec, x, reduction="none").mean(dim=1)
    return (reconst_loss + discrepancy_weight * discrepancy_loss).sum(dim=0)


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
