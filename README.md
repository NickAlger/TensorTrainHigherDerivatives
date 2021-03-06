# TensorTrainFromTensorAction
## Purpose:
1) Construct a tensor train approximation of a tensor from actions of the tensor.
2) Compute the action of higher derivative tensors for quantities of interest that depend on the solution of PDEs.

## Dependencies:
1) Python3
2) numpy
3) scipy
4) collections_extended

For higher derivative PDE stuff:

5) fenics

For generating plots:

6) matplotlib

For compressing the Hilbert tensor with TT-cross for comparison:

7) Matlab
8) TT-Toolbox


## Paper:
Tensor Train Construction from Tensor Actions, with Application to Compression of Large High Order Derivative Tensors.
Nick Alger, Peng Chen, Omar Ghattas

https://arxiv.org/abs/2002.06244


## Primary functionality:
1) Function for constructing tensor train from tensor actions (Section 2 in the paper):
    - tensor_train_from_tensor_action() in tensor_train_from_tensor_action.py

2) Class for computing higher derivative actions (Section 3 in the paper):
    - class HigherDerivativesAction in higher_derivatives_action.py


## Examples:
1) Compressing a Hilbert tensor into tensor train format in:
    - EXAMPLE_hilbert.py

2) Constructing a tensor train Taylor series surrogate model for the noise-whitened parameter-to-output map for a stochastic reaction-diffusion equation with boundary output:
    - EXAMPLE_poisson.py


## Results in the paper are generated by the following files:
1) Figure 3. (Hilbert Tensor):
    - hilbert_tensor_train_cross.m (generate TT-cross data for comparison in Matlab)
    - FIGURE_hilbert.py

2) Table 1. (Mesh Scalability):
    - TABLE_poisson_mesh_scalability.py

3) Figure 4. (Taylor Series Error):
    - FIGURE_poisson_histogram.py
