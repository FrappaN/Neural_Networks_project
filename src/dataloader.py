import torch
import torch_geometric.datasets as datasets
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

def load_mutagenicity(device = 'cpu', seed = None, train_size = 150):
    
    data = datasets.TUDataset(root='../datasets/TUDataset', name='Mutagenicity')


    # shuffle the data using a seed

    seed_everything(seed)
    data = data.shuffle()

    # we ignore the edge attr as in the original paper

    graph_list = [Data(x = graph.x, edge_index = graph.edge_index, y = graph.y, device=device) for graph in data]

    # we split the data into train and test
    if train_size < 1:
        train_size = int(train_size * len(graph_list))

    train_data = graph_list[:train_size]
    test_data = graph_list[train_size:]

    # we create the dataloaders with 16 graphs for batch
    trainloader = DataLoader(train_data, batch_size = 16)
    testloader = DataLoader(test_data, batch_size = 16)

    return trainloader, testloader


