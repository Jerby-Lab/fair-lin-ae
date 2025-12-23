'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

# MLP based Autoencoder 
class AE_MLP(torch.nn.Module):
    """
        An implementation of Multilayer Perceptron (MLP).
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


# Linear Autoencoder
class LinearAE(nn.Module):
    """
        An implementation of Linear Autoencoder
    """
    def __init__(self, input_dim, dimension):
        super(LinearAE, self).__init__()
        self.encoder = nn.Linear(input_dim, dimension)
        self.decoder = nn.Linear(dimension, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded