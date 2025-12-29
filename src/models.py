import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

# MLP based Autoencoder 
class AE_MLP(torch.nn.Module):
    """
    Multilayer Perceptron (MLP)–based autoencoder.

    This module implements a fully connected autoencoder with an arbitrary
    number of hidden layers in the encoder and a symmetric decoder that mirrors
    the encoder architecture. Nonlinear activations are applied between layers,
    except at the output layer.

    Parameters
    ----------
    input_dim : int, default=1024
        Dimensionality of the input features.
    hidden_sizes : tuple of int, default=(256, 64)
        Sizes of successive hidden layers in the encoder. The final entry
        corresponds to the latent (bottleneck) dimension. The decoder mirrors
        this structure in reverse.
    activation : {"relu", "elu"}, default="elu"
        Nonlinear activation function applied after each hidden linear layer
        (except the output layer of the decoder).
    Forward Parameters
    ------------------
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim).

    Returns
    -------
    x_recon : torch.Tensor
        Reconstructed input of shape (batch_size, input_dim).
    hidden_x : torch.Tensor
        Latent representation of shape (batch_size, hidden_sizes[-1]).
    """
    def __init__(self, input_dim=1024, hidden_sizes=(256, 64,), activation='elu'):
        super().__init__()
        if sum(hidden_sizes) > 0: # multi-layer model
            self.hidden_sizes = (input_dim, ) + hidden_sizes
            encoder = []
            for i in range(len(self.hidden_sizes)-1):
                encoder.append(torch.nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1])) 
                if activation=='relu':
                  encoder.append(torch.nn.ReLU())
                elif activation=='elu':
                  encoder.append(torch.nn.ELU())
                else:
                  pass 
            self.encoder = torch.nn.Sequential(*encoder)
            decoder = []
            for i in range(len(self.hidden_sizes)-1,0,-1):
                decoder.append(torch.nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i-1]))
                if i > 1: # we don't use activation on the output layer
                  if activation=='relu':
                    decoder.append(torch.nn.ReLU())
                  elif activation=='elu':
                    decoder.append(torch.nn.ELU())
                  else:
                    pass 
            self.decoder = torch.nn.Sequential(*decoder)

    def forward(self, x):
        if len(self.hidden_sizes) > 1:
            hidden_x = self.encoder(x)
            x = self.decoder(hidden_x)
        return x, hidden_x

class LinearAE(nn.Module):
    """
    Linear autoencoder with a single hidden (latent) layer.

    This module implements a linear encoder–decoder pair with no nonlinear
    activations. When trained with mean squared error, the learned subspace is
    closely related to principal component analysis (PCA).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    dimension : int
        Dimensionality of the latent (bottleneck) space.

    Attributes
    ----------
    encoder : torch.nn.Linear
        Linear mapping from input space to latent space, of shape
        (input_dim → dimension).
    decoder : torch.nn.Linear
        Linear mapping from latent space back to input space, of shape
        (dimension → input_dim).

    Forward Parameters
    ------------------
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim).

    Returns
    -------
    decoded : torch.Tensor
        Reconstructed input of shape (batch_size, input_dim).

    """
    def __init__(self, input_dim, dimension):
        super(LinearAE, self).__init__()
        self.encoder = nn.Linear(input_dim, dimension)
        self.decoder = nn.Linear(dimension, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
