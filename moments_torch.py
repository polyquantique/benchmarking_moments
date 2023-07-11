"""Based on moments_torch_edited.py"""
import torch
from torch.autograd import grad
from torch import tensor, zeros, ones, sqrt, exp, diag, stack, det, inverse
import numpy as np
from scipy.linalg import block_diag
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


def moments_torch(means, cov):
    """Calculates the expectation value of the product of all the number operators of a zero-mean Gaussian states
    with a given covariance matrix"""
    dim = len(cov)
    num_modes = dim//2

    # Set up block-diagonal covariance matrix with multiple two-mode squeezers
    Gamma = tensor(cov)

    dim = Gamma.shape[0]         # Dimension of the covariance matrix
    d = tensor(means)

    # Differentiation parameters y
    ys = [tensor(0., requires_grad=True) for i in range(num_modes)]  


    # Set up the diagonal W matrix and Lambda matrix
    # Note that the W matrix for the moment-generating function depends on 
    # 1 - exp(y) instead of 1 - y for the probability-generating function
    W = diag(stack([1 - exp(y) for y in ys] * 2))

    # Set up the Lambda matrix and the Generating function G
    Lambda = diag(ones(dim)) + W @ (Gamma - diag(ones(dim))) / 2
    G = exp(- d @ (inverse(Lambda) @ (W @ d)) / 2) / sqrt(det(Lambda)) 


    def mixed_first_order_raw_moment():
        deriv = G
        for y in ys:
            deriv = grad(deriv, y, create_graph=True)[0]
        return deriv.detach()

    return mixed_first_order_raw_moment()



def moments_torch_no_displacement(cov):
    """Calculates the expectation value of the product of all the number operators of a zero-mean Gaussian states
    with a given covariance matrix"""
    dim = len(cov)
    num_modes = dim//2

    # Set up block-diagonal covariance matrix with multiple two-mode squeezers
    Gamma = tensor(cov)

    dim = Gamma.shape[0]         # Dimension of the covariance matrix

    # Differentiation parameters y
    ys = [tensor(0., requires_grad=True) for i in range(num_modes)]


    # Set up the diagonal W matrix and Lambda matrix
    # Note that the W matrix for the moment-generating function depends on
    # 1 - exp(y) instead of 1 - y for the probability-generating function
    W = diag(stack([1 - exp(y) for y in ys] * 2))

    # Set up the Lambda matrix and the Generating function G
    Lambda = diag(ones(dim)) + W @ (Gamma - diag(ones(dim))) / 2
    G = 1 / sqrt(det(Lambda))


    def mixed_first_order_raw_moment():
        deriv = G
        for y in ys:
            deriv = grad(deriv, y, create_graph=True)[0]
        return deriv.detach()

    return mixed_first_order_raw_moment()