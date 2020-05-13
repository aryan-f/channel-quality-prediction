import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, rnn_layers=2, rnn_neurons=32):
        super().__init__()
        self.rnn_layers = rnn_layers
        self.rnn_neurons = rnn_neurons
        self.gru = nn.GRU(16, hidden_size=rnn_neurons, num_layers=rnn_layers)
        self.linear = nn.Linear(rnn_neurons, 16)

    def forward(self, interference):
        datapoints, batches, channels = interference.shape
        h0 = torch.randn(self.rnn_layers, batches, self.rnn_neurons)
        output, hn = self.gru(interference, h0)
        output = torch.relu(output[-1, :])  # Latest Output for Each Batch
        output = self.linear(output)
        return torch.sigmoid(output)
