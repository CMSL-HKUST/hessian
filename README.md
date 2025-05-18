# Hessian for Implicit Differentiation 

![Github Star](https://img.shields.io/github/stars/tianjuxue/hessian)
![Github Fork](https://img.shields.io/github/forks/tianjuxue/hessian)
![License](https://img.shields.io/github/license/tianjuxue/hessian)


This repository has two purposes:

- Provides an implementation for Hessian-vector products in implicit differentiable programming
- Shows more examples of differentiable finite elements based on [JAX-FEM](https://github.com/deepmodeling/jax-fem)


## General picture

<p align="middle">
  <img src="images/differentiable_programming.png" width="800" />
</p>
<p align="middle">
    <em >Differentiable programming breaks the boundary between deep learning and differentiable physics.</em>
</p>

This repository is based on [JAX-FEM](https://github.com/deepmodeling/jax-fem) to solve differentiable physics problems, providing second-order derivative information in the form of Hessian-vector products.


## Quick start

Refer to `simple.ipynb` for a simple illustrative example.


## Installation

Works with [JAX-FEM](https://github.com/deepmodeling/jax-fem) version 0.0.9.


## Examples

### E1: Source field identification

*Goal: Change the source term to match observed data.*

<p align="middle">
  <img src="images/example_poisson_inv.png" width="600" />
</p>
<p align="middle">
    <em >Predicted solutions gradually match the reference data.</em>
</p>


### E2: Boundary force identification

*Goal: Change the boundary traction force to match observed displacement.*


<p align="middle">
  <img src="images/example_hyperelasticity_inv.png" width="600" />
</p>
<p align="middle">
    <em >Predicted displacements gradually match the reference displacement.</em>
</p>


### E3: Thermal-mechanical control

*Goal: Change the boundary temperature to achieve desired deformation.*

<p align="middle">
  <img src="images/example_thermal_mechanical_inv.png" width="600" />
</p>
<p align="middle">
    <em >Predicted displacements gradually match the reference displacement.</em>
</p>

### E4: Shape optimization

*Goal: Rotate the square-shaped holes for better beam stiffness.*

<p align="middle">
  <img src="images/example_shape_opt.png" width="600" />
</p>
<p align="middle">
    <em >Predicted displacements gradually match the reference displacement.</em>
</p>


## Paper

Refer to the arXiv version.

## Citations

If you found this library useful in academic or industry work, we appreciate your support if you consider 1) starring the project on Github, and 2) citing relevant papers.

