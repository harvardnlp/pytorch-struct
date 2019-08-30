
# Pytorch-Struct

[![Build Status](https://travis-ci.org/harvardnlp/pytorch-struct.svg?branch=master)](https://travis-ci.org/harvardnlp/pytorch-struct)
[![Coverage Status](https://coveralls.io/repos/github/harvardnlp/pytorch-struct/badge.svg?branch=master)](https://coveralls.io/github/harvardnlp/pytorch-struct?branch=master)

<p align="center">
  <img src="https://github.com/harvardnlp/pytorch-struct/raw/master/download.png">
  </p>



A library of tested, GPU implementations of core structured prediction algorithms for deep learning applications. 
(or an implementation of <a href="https://www.cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf">Inside-Outside and Forward-Backward Algorithms Are Just Backprop"<a/>)


## Getting Started

```
pip install . 
```

```python
import torch_struct
import torch
batch, N = 10,  100
scores = torch.rand(N, 100, 100, requires_grad=True)

# Tree marginals
marginals = torch.deptree(scores)

# Tree Argmax
argmax = torch.deptree(scores, seminring=torch_struct.MaxSemiring)
max_score = torch.mul(argmax, scores)

# Tree Counts
ones = torch.ones(N, 100, 100)
ntrees = torch.deptree(ones, semiring=torch_struct.StdSemiring)

# Tree Sample
sample = torch.deptree(scores, seminring=torch_struct.SampledSemiring)

# Tree Partition
v, _ = torch.deptree_inside(scores)

# Tree Max
v, _ = torch.deptree_inside(scores, semiring=torch_struct.MaxSemiring)

```

## Library

Current algorithms implemented:

* Linear Chain (CRF / HMM)
* Semi-Markov (CRF / HSMM)
* Dependency Parsing (Projective and Non-Projective)
* CKY (CFG)

Design Strategy:

1) Minimal implementatations. Most are 10 lines.
2) Batched for GPU.
3) Code can be ported to other backends

Semirings:

* Log Marginals
* Max and MAP computation
* Sampling through specialized backprop

Example: https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/Examples.ipynb

# Applications

Application Example (to come):

* Structured Attention
* EM training
* Stuctured VAE
* Posterior Regularization
