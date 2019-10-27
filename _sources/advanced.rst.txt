


============================
Advanced Usage: Semirings
============================

All of the distributional code is implemented through a series of
semiring objects. These are passed through dynamic programming
backends to compute the distributions.


Standard Semirings
===================

.. autoclass:: torch_struct.LogSemiring
.. autoclass:: torch_struct.StdSemiring
.. autoclass:: torch_struct.MaxSemiring

Higher-Order Semirings
=========================
.. autoclass:: torch_struct.EntropySemiring

Sampling Semirings
===================

.. autoclass:: torch_struct.SampledSemiring
.. autoclass:: torch_struct.MultiSampledSemiring


Dynamic Programming
===================

.. autoclass:: torch_struct.LinearChain
.. autoclass:: torch_struct.SemiMarkov
.. autoclass:: torch_struct.DepTree
.. autoclass:: torch_struct.CKY
