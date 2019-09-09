# Pytorch-Struct

[![Build Status](https://travis-ci.org/harvardnlp/pytorch-struct.svg?branch=master)](https://travis-ci.org/harvardnlp/pytorch-struct)
[![Coverage Status](https://coveralls.io/repos/github/harvardnlp/pytorch-struct/badge.svg?branch=master)](https://coveralls.io/github/harvardnlp/pytorch-struct?branch=master)

<p align="center">
  <img src="https://github.com/harvardnlp/pytorch-struct/raw/master/download.png">
  </p>



A library of tested, GPU implementations of core structured prediction algorithms for deep learning applications.
(or an implementation of <a href="https://www.cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf">Inside-Outside and Forward-Backward Algorithms Are Just Backprop"<a/>)


## Getting Started


```python
!pip install -qU git+https://github.com/harvardnlp/pytorch-struct
!pip install -q matplotlib
```


```python
import torch
from torch_struct import DepTree, LinearChain, MaxSemiring, SampledSemiring
import matplotlib.pyplot as plt
def show(x): plt.imshow(x.detach())
```


```python
# Make some data.
vals = torch.zeros(2, 10, 10) + 1e-5
vals[:, :5, :5] = torch.rand(5)
vals[:, 5:, 5:] = torch.rand(5) 
vals = vals.log()
show(vals[0])
```


![png](README_files/README_4_0.png)



```python
# Compute marginals
marginals = DepTree().marginals(vals)
show(marginals[0])
```


![png](README_files/README_5_0.png)



```python
# Compute argmax
argmax = DepTree(MaxSemiring).marginals(vals)
show(argmax.detach()[0])
```


![png](README_files/README_6_0.png)



```python
# Compute scoring and enumeration (forward / inside)
log_partition = DepTree().sum(vals)
max_score = DepTree(MaxSemiring).sum(vals)
max_score = DepTree().score(argmax, vals)
```


```python
# Compute samples 
sample = DepTree(SampledSemiring).marginals(vals)
show(sample.detach()[0])
```


![png](README_files/README_8_0.png)



```python
# Padding/Masking built into library.
marginals = DepTree().marginals(
    vals,
    lengths=torch.tensor([10, 7]))
show(marginals[0])
plt.show()
show(marginals[1])
```


![png](README_files/README_9_0.png)



![png](README_files/README_9_1.png)



```python
# Many other structured prediction approaches
chain = torch.zeros(2, 10, 10, 10) + 1e-5
chain[:, :, :, :] = vals.unsqueeze(-1).exp()
chain[:, :, :, :] += torch.eye(10, 10).view(1, 1, 10, 10) 
chain[:, 0, :, 0] = 1
chain[:, -1,9, :] = 1
chain = chain.log()

marginals = LinearChain().marginals(chain)
show(marginals.detach()[0].sum(-1))
```


![png](README_files/README_10_0.png)


## Library

Current algorithms implemented:

* Linear Chain (CRF / HMM)
* Semi-Markov (CRF / HSMM)
* Dependency Parsing (Projective and Non-Projective)
* CKY (CFG)

* Integration with `torchtext` and `pytorch-transformers` 

Design Strategy:

1) Minimal implementatations. Most are 10 lines.
2) Batched for GPU.
3) Code can be ported to other backends

Semirings:

* Log Marginals
* Max and MAP computation
* Sampling through specialized backprop




## Examples

* BERT <a href="https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/BertTagger.ipynb">Part-of-Speech</a> 
* BERT <a href="https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/BertDependencies.ipynb">Dependency Parsing</a>
* <a href="https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/Unsupervised_CFG.ipynb">Unsupervised Learning </a>
* Structured VAE (to come)
* Structured attention (to come)
