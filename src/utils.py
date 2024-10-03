from torch_geometric.data import Data
import numpy as np
import torch
from tqdm import tqdm


# We define a perm_graph function to permute the nodes of a graph

def perm_graph(data, perm):
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    batch = data.batch.clone()
    y = data.y.clone()

    x = x[perm]
    inv_perm = np.argsort(perm)
    edge_index[0] = inv_perm[edge_index[0]]
    edge_index[1] = inv_perm[edge_index[1]]
    batch = batch[perm]

    new_data = Data(x=x, edge_index=edge_index, batch=batch, y=y)
    return new_data

# The equivariance_robustness function computes the node-wise scores for the equivariance robustness metric
# for n_test random permutations of the nodes of the graph, we compare the permuted original explanation
# with the explanation of the permuted graph using a cosine similarity metric

def equivariance_robustness(loader, explainer, ntest=50):
    ntest = 50
    node_wise_scores = np.zeros((len(loader), ntest))

    for k in tqdm(range(ntest)):
        
        for i, data in enumerate(loader):    
            original_attr = explainer.explain(data)
            original_attr_normalized = original_attr / np.sqrt((original_attr**2).sum())

            perm = torch.randperm(data.num_nodes)
            data_perm = perm_graph(data, perm)

            attr = explainer.explain(data_perm)

            attr_normalized = attr / np.sqrt((attr**2).sum())

            # cosine similarity between the normalized, flattened arrays.

            node_wise_scores[i, k] = (original_attr_normalized[perm].flatten()*attr_normalized.flatten()).sum()

    # the result are average over the batches in dataloader
    return node_wise_scores.mean(axis=0)

