import torch.nn as nn

class DynamicANN(nn.Module):
    def __init__(self, input_dim, layers, dropout_rate=0.2):
        super(DynamicANN, self).__init__()
        modules = []

        prev_dim = input_dim
        for layer_size in layers:
            modules.append(nn.Linear(prev_dim, layer_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout_rate))
            prev_dim = layer_size

        modules.append(nn.Linear(prev_dim, 1))  # single output for regression

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
