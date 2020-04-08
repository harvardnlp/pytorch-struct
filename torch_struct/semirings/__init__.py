from .semirings import (
    LogSemiring,
    StdSemiring,
    KMaxSemiring,
    MaxSemiring,
    EntropySemiring,
    TempMax
)

from .fast_semirings import FastLogSemiring, FastMaxSemiring, FastSampleSemiring


from .checkpoint import CheckpointSemiring, CheckpointShardSemiring

from .sparse_max import SparseMaxSemiring

from .sample import MultiSampledSemiring, SampledSemiring, GumbelSoftmaxSemiring


# For flake8 compatibility.
__all__ = [
    FastLogSemiring,
    FastMaxSemiring,
    FastSampleSemiring,
    GumbelSoftmaxSemiring,
    LogSemiring,
    StdSemiring,
    SampledSemiring,
    MaxSemiring,
    SparseMaxSemiring,
    KMaxSemiring,
    EntropySemiring,
    MultiSampledSemiring,
    CheckpointSemiring,
    CheckpointShardSemiring,
    TempMax,
]
