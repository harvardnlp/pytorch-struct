# Torch-Struct: Structured Prediction Library 


![Tests](https://github.com/harvardnlp/pytorch-struct/workflows/Tests/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/harvardnlp/pytorch-struct/badge.svg?branch=master)](https://coveralls.io/github/harvardnlp/pytorch-struct?branch=master)

<p align="center">
  <img src="https://github.com/harvardnlp/pytorch-struct/raw/master/download.png">
  </p>


A library of tested, GPU implementations of core structured prediction algorithms for deep learning applications.

* HMM / LinearChain-CRF 
* HSMM / SemiMarkov-CRF 
* Dependency Tree-CRF 
* PCFG Binary Tree-CRF 
* ...

Designed to be used as efficient batched layers in other PyTorch code. 

[Tutorial paper](https://arxiv.org/abs/2002.00876) describing methodology.

## Getting Started


```python
!pip install -qU git+https://github.com/harvardnlp/pytorch-struct
# Optional CUDA kernels for FastLogSemiring
!pip install -qU git+https://github.com/harvardnlp/genbmm
# For plotting.
!pip install -q matplotlib
```


```python
import torch
from torch_struct import DependencyCRF, LinearChainCRF
import matplotlib.pyplot as plt
def show(x): plt.imshow(x.detach())
```


```python
# Make some data.
vals = torch.zeros(2, 10, 10) + 1e-5
vals[:, :5, :5] = torch.rand(5)
vals[:, 5:, 5:] = torch.rand(5) 
dist = DependencyCRF(vals.log())
show(dist.log_potentials[0])
```


![png](README_files/README_4_0.png)



```python
# Compute marginals
show(dist.marginals[0])
```


![png](README_files/README_5_0.png)



```python
# Compute argmax
show(dist.argmax.detach()[0])
```


![png](README_files/README_6_0.png)



```python
# Compute scoring and enumeration (forward / inside)
log_partition = dist.partition
max_score = dist.log_prob(dist.argmax)
```


```python
# Compute samples 
show(dist.sample((1,)).detach()[0, 0])
```


![png](README_files/README_8_0.png)



```python
# Padding/Masking built into library.
dist = DependencyCRF(vals, lengths=torch.tensor([10, 7]))
show(dist.marginals[0])
plt.show()
show(dist.marginals[1])
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

dist = LinearChainCRF(chain)
show(dist.marginals.detach()[0].sum(-1))
```


![png](README_files/README_10_0.png)


## Library

Full docs: http://nlp.seas.harvard.edu/pytorch-struct/

Current distributions implemented:

* LinearChainCRF 
* SemiMarkovCRF 
* DependencyCRF 
* NonProjectiveDependencyCRF
* TreeCRF 
* NeuralPCFG / NeuralHMM

Each distribution includes: 

* Argmax, sampling, entropy, partition, masking, log_probs, k-max

Extensions:

* Integration with `torchtext`, `pytorch-transformers`, `dgl`
* Adapters for generative structured models (CFG / HMM / HSMM)
* Common tree structured parameterizations TreeLSTM / SpanLSTM


## Low-level API: 

Everything implemented through semiring dynamic programming. 

* Log Marginals
* Max and MAP computation
* Sampling through specialized backprop
* Entropy and first-order semirings. 


## Examples

* BERT <a href="https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/BertTagger.ipynb">Part-of-Speech</a> 
* BERT <a href="https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/BertDependencies.ipynb">Dependency Parsing</a>
* <a href="https://github.com/harvardnlp/pytorch-struct/blob/master/notebooks/Unsupervised_CFG.ipynb">Unsupervised Learning</a> 
* <a href="https://github.com/harvardnlp/pytorch-struct/blob/master/examples/tree.py">Structured VAE </a>

<img src="https://media.giphy.com/media/IdxKpsWBHa5PpjuhHM/giphy.gif">



## Citation

```
@misc{alex2020torchstruct,
    title={Torch-Struct: Deep Structured Prediction Library},
    author={Alexander M. Rush},
    year={2020},
    eprint={2002.00876},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

This work was partially supported by NSF grant IIS-1901030. 
