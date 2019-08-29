<center><img src="https://github.com/harvardnlp/pytorch-struct/raw/master/download.png"></center>

# Pytorch-Struct


A library of tested, GPU implementations of core structured prediction algorithms for deep learning applications.

## Install

## Library

Implemented:

* Linear Chain (CRF / HMM)
* Semi-Markov (CRF / HSMM)
* Dependency Parsing (Projective and Non-Projective)
* CKY (CFG)

Design Strategy:

1) Minimal code. Entire library is around 200 loc.
2) All algorithms are implemented for GPU.
3) Only implement forward pass, use gradients for marginals.

Semirings with Gradients

* Log Marginals
* Max and MAP computation
* Sampling through specialized backprop

Example: https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/Examples.ipynb

Application Example (to come):

* Structured Attention
* EM training
* Stuctured VAE
* Posterior Regularization
