import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(32, 32), hidden_nonlinearity=nn.ReLU, output_nonlinearity=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        layers = nn.ModuleList()
        in_dim = input_dim
        for next_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, next_dim))
            if hidden_nonlinearity is not None:
                layers.append(hidden_nonlinearity)
            in_dim = next_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if output_nonlinearity is not None:
            layers.append(output_nonlinearity)
        
        self.net = nn.Sequential(*layers)

        # Initialize weights with xavier_uniform
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)
