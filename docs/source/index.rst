
PyTorch-Struct
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
============

A library for structured prediction in pytorch.


Distributional Interface
========================

.. autoclass:: torch_struct.StructDistribution
   :members:

Linear Chain
--------------

.. autoclass:: torch_struct.LinearChainCRF


Semi-Markov
--------------

.. autoclass:: torch_struct.SemiMarkovCRF


Dependency Tree
----------------

.. autoclass:: torch_struct.DependencyCRF


Binary Tree
--------------

.. autoclass:: torch_struct.TreeCRF

Context-Free Grammar
---------------------

.. autoclass:: torch_struct.SentCFG







Networks
===========

Common structured networks.


.. autoclass:: torch_struct.networks.TreeLSTM

.. autoclass:: torch_struct.networks.NeuralCFG

.. autoclass:: torch_struct.networks.SpanLSTM


Data
====

Datasets for common structured prediction tasks.

.. autoclass:: torch_struct.data.ConllXDataset
.. autoclass:: torch_struct.data.ListOpsDataset


Advanced Usage: Semirings
=========================

Standard Semirings
------------------

.. autoclass:: torch_struct.LogSemiring
.. autoclass:: torch_struct.StdSemiring
.. autoclass:: torch_struct.MaxSemiring

Higher-Order Semirings
----------------------
.. autoclass:: torch_struct.EntropySemiring

Sampling Semirings
----------------------

.. autoclass:: torch_struct.SampledSemiring
.. autoclass:: torch_struct.MultiSampledSemiring


Dynamic Programming
-------------------

.. autoclass:: torch_struct.LinearChain
.. autoclass:: torch_struct.SemiMarkov
.. autoclass:: torch_struct.DepTree
.. autoclass:: torch_struct.CKY



References
==========

.. bibliography:: refs.bib


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
