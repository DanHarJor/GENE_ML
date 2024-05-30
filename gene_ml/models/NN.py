import torch
from torch import nn
from .base import Model

class NN(Model, nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(8,30)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(30,2)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x