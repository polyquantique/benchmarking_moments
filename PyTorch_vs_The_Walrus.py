#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 4 20:08:14 2023

@author: Erik Fitzke

This code computes the expectation value of the product of number operators
for the modes of a Gaussian state with zero displacement.

Two computation methods are compared:

-   Using the function photon_number_moment() from The Walrus
    https://the-walrus.readthedocs.io
    implementing a method evaluating the Hafnian function as described in
    Cardin, Quesada (2022):
    'Photon-number moments and cumulants of Gaussian states'
    https://doi.org/10.48550/arXiv.2212.06067

-   Using PyTorch for automatic differentiation of the
    multivariate moment generating function as described in
    E. Fitzke et al., APL Photonics 8, 026106 (2023)
    https://doi.org/10.1063/5.0129638

    This code is based on the code for the computation of the photon
    number distribution presented in the technical report
    https://doi.org/10.26083/tuprints-00023061.

For the demonstration, a state consisting of a number of two-mode squeezed
vacuum states is considered.

Tested with The Walrus 0.20.0 and PyTorch 1.11.0.

"""

import torch
from torch.autograd import grad
from torch import tensor, ones, sqrt, exp, diag, stack, det
import numpy as np
from scipy.linalg import block_diag
from thewalrus.quantum import photon_number_moment


mean_pair_no_per_squeezer = 2  # Mean number of photon pairs per squeezer
num_squeezers = 3              # Number of equally strong squeezers


# Set default type of torch variables and tensors to 64-bit floating point
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


###############################################################################
######    Set up covariance matrix of TMSV with squeezing angle = zero   ######
###############################################################################

r = np.arcsinh(np.sqrt(mean_pair_no_per_squeezer))  # TMSV squeezing parameter
c, s = np.cosh(2*r), np.sinh(2*r)

covmat_TMSV = np.array([[c,  s,  0,  0],
                        [s,  c,  0,  0],
                        [0,  0,  c, -s],
                        [0,  0, -s,  c]])  # Covariance of a two-mode squeezer

mode_indices = range(num_squeezers * 2)

moment_orders = {i: 1 for i in mode_indices}

# Set up block-diagonal covariance matrix with multiple two-mode squeezers
Gamma = tensor(block_diag(*([covmat_TMSV] * num_squeezers)))

dim = Gamma.shape[0]         # Dimension of the covariance matrix

# Create differentiation parameters y
ys = [tensor(0., requires_grad=True) for i in mode_indices]

# Set up the diagonal W matrix, consisting of two diagonal blocks
# The W matrix for the moment-generating function depends on
# 1 - exp(y) instead of 1 - y for the probability-generating function,
# which can be used to compute the photon number distribution.
W = diag(stack([1 - exp(y) for y in ys] * 2))

# Set up the Lambda matrix
Lambda = diag(ones(dim)) + W @ (Gamma - diag(ones(dim))) / 2

# Set up the the generating function G for a Gaussian state with displacement
# vector d = 0.
G = 1 / sqrt(det(Lambda))


def mixed_first_order_raw_moment():
    deriv = G
    for y in ys:
        deriv = grad(deriv, y, create_graph=True)[0]
    return np.float64(deriv.detach())


###############################################################################
######           Evaluate the product of the number operators            ######
###############################################################################

print(f'Compute the mixed first-order raw moment for {num_squeezers} two-mode'
      ' squeezers,\neach with a mean photon pair number of'
      f' {mean_pair_no_per_squeezer}.')

print('Result using The Walrus:', photon_number_moment(np.zeros(len(Gamma)),
                                                       np.array(Gamma),
                                                       moment_orders))

print('Result using pyTorch:', mixed_first_order_raw_moment())
