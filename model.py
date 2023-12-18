import torch
import torch.nn as nn

class SimpleMLP(torch.nn.Module):
    def __init__(self, n_in=768, hid1=64, hid2=32, n_out=5, activation=nn.ReLU()):
        super(SimpleMLP, self).__init__()

        self.linear_in = nn.Linear(n_in, hid1)
        self.linear_mid = nn.Linear(hid1, hid2)
        self.linear_out = nn.Linear(hid2, n_out)
        self.activation = activation
       
    def forward(self, x):
        return self.linear_out(self.activation(self.linear_mid(self.activation(self.linear_in(x)))))
    
class SimplerMLP(torch.nn.Module):
    def __init__(self, n_in=768, hid1=8, n_out=5, activation=nn.ReLU()):
        super(SimplerMLP, self).__init__()

        self.linear_in = nn.Linear(n_in, hid1)
        self.linear_out = nn.Linear(hid1, n_out)
        self.activation = activation
       
    def forward(self, x):
        return self.linear_out(self.activation(self.linear_in(x)))