import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
import logging
import torch.nn.init as init
from model.mlp import MLP


class MLP_2(nn.Module):
    """
    Expert network class. Using Tanh as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - hidden_size (int): The size of the hidden layer.
    
    MLP_2 need parameter for mlp2 output_size2=2
    """
    def __init__(self, args):
        super().__init__()
        self.depth = getattr(args, "mlp_depth", 1)
        input_size = getattr(args, "input_size", 2)
        hidden_size = getattr(args, "mlp_hidden_size", 80)
        hidden_size2=getattr(args,"hidden_size2",200)
        output_size = getattr(args, "output_size", 1)
        output_size2=getattr(args,"output_size2",2)
        

        self.activation = nn.Tanh()
        self.mlp1=MLP(args)  #u
        mlp2_layers = []
        mlp2_layers.append(nn.Linear(input_size, hidden_size2))
        mlp2_layers.append(self.activation)
        for _ in range(self.depth - 1):
            mlp2_layers.append(nn.Linear(hidden_size2, hidden_size2))
            mlp2_layers.append(self.activation)
        
        mlp2_layers.append(nn.Linear(hidden_size2, output_size2,bias=False))

        self.mlp2 = nn.Sequential(*mlp2_layers)

        self._init_weights()
        self._report_trainable()

    def _report_trainable(self):
        total = 0
        print("=== MLP_2 Trainable parameters ===")
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += p.numel()
        print(f"MLP_2 trainable params: {total}")

    def _init_weights(self):
        for m in self.mlp2:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, y):
        u = self.mlp1(y)
        q = self.mlp2(y)
        return u, q
        

    
