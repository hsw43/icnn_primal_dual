# A primal-dual algorithm for image reconstruction with input-convex neural network regularizers

This repo contains python scripts for implementating a primal dual algorithm for solving variational problems associated with input-convex neural network (ICNN) regularizer. It solves problems of the form

$$\min_{x}D(Ax,y)+R_{\theta}(x),$$

where $A$ and $D$ are the forward operator and data fidelity respectively, and $R_{\theta}$ is a pretrained ICNN-based regularizer.

For a detailed description of the algorithm and theoretical results, see: [https://arxiv.org/abs/2008.02839](https://arxiv.org/abs/2410.12441).

## Examples

The numerical examples included are:

* [example_denoise.py](example_denoise.py) Variational reconstruction for salt and pepper denoising (L1 fidelity). 
* [example_inpaint.py](example_inpaint.py) Variational reconstruction for inpainting (Gaussian noise, L2 fidelity). 
* [example_ct.py](example_ct.py) Variational reconstruction for CT (Poisson noise, KL fidelity).
* [example_ct_deep.py](example_ct.py) Variational reconstruction for CT with a deeper ICNN regularizer (Poisson noise, KL fidelity).
* [example_ar_training.py](example_ar_training.py) Example adversarial training (For inpainting). Script adapted from this [repo](https://github.com/Subhadip-1/data_driven_convex_regularization).
* [example_bilevel_training.py](example_bilevel_training.py) Example bilevel training (For inpainting).

The CT examples are based on [ASTRA](https://www.astra-toolbox.com/) to compute the line integrals in the forward operator.
- To run the CT examples, create a conda environment with the required dependencies by `conda env create -f environment_ct.yml`.

## References

[1] Ehrhardt, Mukherjee & Wong (2025). [A primal-dual algorithm for image reconstruction with input-convex neural network regularizers](https://arxiv.org/abs/2410.12441). Accepted in SIAM Journal on Imaging Sciences.

[2] Mukherjee, Dittmer, Shumaylov, Lunz, Öktem & Schönlieb (2021). [Learned convex regularizers for inverse problems](https://arxiv.org/abs/2008.02839).
