# Neural_Networks_project

Partial re-implementation of the paper 'Evaluating the Robustness of Interpretability Methods through Explanation Invariance and Equivariance'

The paper is a benchmark of the robustness of the explainers under symmetrical transformation. In this re-implementation I focused on replicating the results on Graph Neural Networks and attribution explantions, by checking that the explanation produced by the explainer are equivariant under node permutation. Moreover, I also reproduce the proposed method to enforce equivariance in any attribution.

The experiment.ipynb notebook in the /notebooks folder shows how to use the re-implementation of the method, while the actual code that I re-implemented can be found in the /src folder. The requirements of the conda environment I used are in the requirements.txt file.

The file /src/captumexplainer.py is mostly a copying and paste from the homonym class in Pytorch Geometric, where I added support for the GradientSHAP explainer.

The GitHub Copilot LLM was active while coding, and was used in auto-complete mode.
