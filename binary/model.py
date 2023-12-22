import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(self, n_in, layer_sizes=[768, 64, 32, 5], n_out=5, activation=nn.ReLU()):
        super(MLP, self).__init__()

        if layer_sizes[0] != n_in or layer_sizes[-1] != n_out:
            raise ValueError("Layer sizes must include in and output sizes.")
        
        layers = []
        for s_in, s_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(s_in, s_out))
            if s_out != n_out:
                layers.append(activation)
        self.model = nn.Sequential(*layers)
       
    def forward(self, x):
        return self.model(x)