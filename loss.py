import torch
from torch.distributions import Normal
from torch.nn import NLLLoss
from torch.distributions import kl_divergence
from torch.nn import functional
from torch.nn import CrossEntropyLoss


def loss_function_vae(dec, x, mu, stdev, kl_weight=1.0):
    # sum over genes, mean over samples, like trvae
    
    mean = torch.zeros_like(mu) # tensor with the same shape as mu but filled with zeros
    scale = torch.ones_like(stdev) #  tensor with the same shape as stdev but filled with ones

    KLD = kl_divergence(Normal(mu, stdev), Normal(mean, scale)).mean(dim=1) # Regularisation term

    reconst_loss = functional.mse_loss(dec, x, reduction='none').mean(dim=1)
    print('KLD loss term :', KLD)
    print('Reconstruction loss : ', reconst_loss)
    return (reconst_loss + kl_weight * KLD).sum(dim=0)
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

def gaussian_kernel(a, b, kernel_length=1.0, device='cpu'):
    x = a.view(len(a), -1)
    y = b.view(len(b), -1)
    dim = x.size(1)
    x = x.unsqueeze(1)  # shape: (len(a), 1, dim)
    y = y.unsqueeze(0)  # shape: (1, len(b), dim)
    kernel = torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * kernel_length ** 2))
    return kernel

def compute_mmd_loss(z, device='cpu'):
    true_samples = torch.randn(z.size(0), z.size(1), device=device)
    zz_kernel = gaussian_kernel(z, z, device=device)
    tt_kernel = gaussian_kernel(true_samples, true_samples, device=device)
    zt_kernel = gaussian_kernel(z, true_samples, device=device)
    mmd = zz_kernel.mean() + tt_kernel.mean() - 2 * zt_kernel.mean()
    return mmd


# Francesco's version


def loss_function_classification(pred, target, class_weights):
    """
    Compute cross-entropy loss.

    :param pred: predictions (logits before softmax)
    :param target: actual classes (as indices, not one-hot encoded)
    :param class_weights: weights - one per class
    :return: Weighted prediction loss. Summed for all the samples, not averaged.
    """
    loss_function = CrossEntropyLoss(weight=class_weights, reduction="none")
    # Assuming target is one-hot encoded, converting to class indices
    target_indices = torch.argmax(target, dim=1)
    return loss_function(pred, target_indices).sum()
