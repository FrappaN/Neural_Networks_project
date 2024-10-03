
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
import torch_geometric.nn.aggr as aggr


# Define the GNN model
# the default parameters are the ones used in the original paper

class MUTAG_GNN(nn.Module):
    def __init__(self, hidden_dim = 32, num_layers = 5, dropout = 0.5):
        super(MUTAG_GNN, self).__init__()
        
        self.conv1 = gnn.GraphConv(14, hidden_dim)

        self.convs = nn.ModuleList([gnn.GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])

        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layers+1)])

        self.pool = aggr.SumAggregation()
        
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, 2)
        self.out = nn.LogSoftmax(dim = 1)

        self.criterion = torch.nn.NLLLoss()
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relus[0](x)
        for conv, relu in zip(self.convs, self.relus[1:]):
            x = conv(x, edge_index)
            x = relu(x)
        x = self.pool(x, batch)
        x = self.lin1(x)
        x = self.relus[-1](x)
        x = self.dropout(x)
        x = self.lin2(x)
        return self.out(x)
    
    def latent(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relus[0](x)
        for conv, relu in zip(self.convs, self.relus[1:]):
            x = conv(x, edge_index)
            x = relu(x)
        x = self.pool(x, batch)
        x = self.lin1(x)
        x = self.relus[-1](x)
        return x
    
