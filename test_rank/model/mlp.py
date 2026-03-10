import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
import logging
import torch.nn.init as init


class MLP(nn.Module):
    """
    Expert network class. Using Tanh as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - hidden_size (int): The size of the hidden layer.
    """
    def __init__(self, args):
        super().__init__()
        self.depth = getattr(args, "mlp_depth", 6)
        input_size = getattr(args, "input_size", 2)
        hidden_size = getattr(args, "mlp_hidden_size", 40)
        output_size = getattr(args, "output_size", 1)

        self.activation = nn.Tanh()

        # ===== Feature extractor =====
        feature_layers = []
        feature_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(self.depth - 1):
            feature_layers.append(nn.Linear(hidden_size, hidden_size))
        self.feature = nn.ModuleList(feature_layers)

        # ===== Output head =====
        self.head = nn.Linear(hidden_size, output_size,bias=False)

        self._init_weights()
        self._report_trainable()
        
    def _report_trainable(self):
            total = 0
            print("=== MLP Trainable parameters ===")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    n = p.numel()
                    total += n
            print(f"MLP trainable params: {total}")
            
    def _init_weights(self):
        for m in self.feature:
            if isinstance(m, nn.Linear):
                # Xavier uniform initialization for weights
                init.xavier_uniform_(m.weight)
                # Zero initialization for biases
                init.zeros_(m.bias)
        
        # Initialize the output head (Linear layer)
        if isinstance(self.head, nn.Linear):
            init.xavier_uniform_(self.head.weight)
            
        
    # @torch.no_grad()
    def forward_penultimate(self, x):
        """
        Return the activation after the last hidden layer (before final linear).
        Shape: [N, hidden_size]
        """
        h = x
        for layer in self.feature:
            h = self.activation(layer(h))
        return h
    def forward(self, y):
        for i, layer in enumerate(self.feature):
            y = layer(y)
            y = self.activation(y)
        y = self.head(y)
        return y
    
# feature getter
# @torch.no_grad()
def mlp_penultimate_getter(model, x):
    """
    Get the penultimate layer activations from an MLP model.

    Parameters:
    - model (MLP): The MLP model instance.
    - x (torch.Tensor): Input tensor of shape [N, input_size].

    Returns:
    - torch.Tensor: Activations from the penultimate layer of shape [N, hidden_size].
    """
    return model.forward_penultimate(x)