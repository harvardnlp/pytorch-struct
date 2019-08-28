# Pytorch-Struct

<img src="im.png">

A library of highly-tested, GPU-ready implementations of core structured prediction algorithms. 

## Install

## Library

Implemented: 

* Linear Chain CRF / HMM 
* Semi-Markov CRF / HSMM
* Dependency Parsing 
* CKY / Context-Free Grammars

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
