import numpy as np
import torch
from graphxai.explainers import GraphLIME
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import PGExplainer, GNNExplainer
from captumexplainer import CaptumExplainer
from utils import perm_graph


# I created a class for each explainer with a unified interface
# I re-implement the CaptumExplainer class from Pytorch Geometric to include the GradientShap method,
# which is not included in the library
# The choice of baselines and parameters follows the original implementation of the paper.

class gnnexp_scores():
    def __init__(self, model, device='cpu', **kwargs):
        self.device = device
        self.explainer_name = 'GNNExplainer'
        model_config = ModelConfig(mode='multiclass_classification',task_level='graph', return_type='raw')
        self.model = model
        self.explainer = Explainer(
            self.model,  # It is assumed that model outputs a single tensor.
            algorithm=GNNExplainer(epochs=200, lr=0.001).to(self.device),
            explanation_type='phenomenon',
            node_mask_type='attributes',
            edge_mask_type=None,
            model_config = model_config,
        )
        return

    def explain(self, data ):

        explanation = self.explainer(data.x, data.edge_index, batch=data.batch, target=data.y)
        scores_features = explanation.node_mask.detach().cpu().numpy()

        return scores_features


class captum_scores():
    def __init__(self, model, explainer_name, **kwargs):
        self.explainer_name = explainer_name
        self.model = model
        if 'baseline_type' in kwargs:
            self.baseline_type = kwargs['baseline_type']
        else:
            self.baseline_type = None
        return


    def explain(self, data):

        model_config = ModelConfig(mode='multiclass_classification', task_level='graph', return_type='probs')
        explainer = Explainer(
            self.model,  # It is assumed that model outputs a single tensor.
            algorithm=CaptumExplainer(attribution_method=self.explainer_name, baseline=self.baseline_type),
            explanation_type='phenomenon',
            node_mask_type='attributes',
            edge_mask_type=None,
            model_config = model_config,
        )

        explanation = explainer(data.x, data.edge_index, batch=data.batch, target=data.y)
        scores_features = explanation.node_mask.detach().cpu().numpy()

        return scores_features


class equivariant_explainer():
    def __init__(self, model, original_explainer, n_perms=50, **kwargs):
        self.original_explainer = original_explainer
        self.model = model
        self.n_perms = n_perms
        return
    
    def explain(self, data):

        attr = self.original_explainer.explain(data)

        for _ in range(self.n_perms-1):

            perm = torch.randperm(data.num_nodes)
            inv_perm = torch.argsort(perm)

            data_perm = perm_graph(data, perm)

            # for each permutation we compute the explanation
            curr_attr = self.original_explainer.explain(data_perm)

            # the attribution must be permuted back to the original nodes order
            curr_attr = curr_attr[inv_perm]

            attr += curr_attr

        # the final attribution is the average of the n_perms attributions

        return attr / (self.n_perms+1)
    
