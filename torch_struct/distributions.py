import torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property
from .linearchain import LinearChain
from .semimarkov import SemiMarkov
from .deptree import DepTree
from .cky_crf import CKY_CRF
from .semirings import (
    LogSemiring,
    MaxSemiring,
    StdSemiring,
    SampledSemiring,
    EntropySemiring,
    MultiSampledSemiring,
)

class StructDistribution(Distribution):
    has_enumerate_support = True

    def __init__(self, log_potentials, lengths=None):
        batch_shape = log_potentials.shape[:1]
        event_shape = log_potentials.shape[1:]
        self.log_potentials = log_potentials
        self.lengths = lengths
        super().__init__(batch_shape=batch_shape,
                         event_shape=event_shape)

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    # @constraints.dependent_property
    # def support(self):
    #     pass

    # @property
    # def param_shape(self):
    #     return self._param.size()

    @lazy_property
    def partition(self):
        return self.struct(LogSemiring).sum(self.log_potentials, self.lengths)

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def sample(self, sample_shape=torch.Size()):
        assert(len(sample_shape) == 1)
        nsamples = sample_shape[0]
        samples = []
        for k in range(nsamples):
            if k % 10 == 0:
                sample = self.struct(MultiSampledSemiring).marginals(self.log_potentials,
                                                                lengths=self.lengths)
                sample = sample.detach()
            tmp_sample = MultiSampledSemiring.to_discrete(sample, (k % 10) +1)
            samples.append(tmp_sample)
        return torch.stack(samples)


    def log_prob(self, value):
        d = value.dim()
        batch_dims = range(d - len(self.event_shape))
        v = self.struct().score(self.log_potentials, value.type_as(self.log_potentials), batch_dims=batch_dims)
        return v - self.partition

    @lazy_property
    def entropy(self):
        return self.struct(EntropySemiring).sum(self.log_potentials, self.lengths)

    @lazy_property
    def argmax(self):
        return self.struct(MaxSemiring).marginals(self.log_potentials, self.lengths)

    @lazy_property
    def marginals(self):
        return self.struct(LogSemiring).marginals(self.log_potentials, self.lengths)

    def enumerate_support(self, expand=True):
         _, _, edges, enum_lengths = self.struct().enumerate(self.log_potentials, self.lengths)
         # if expand:
         #     edges = edges.unsqueeze(1).expand(edges.shape[:1] + self.batch_shape[:1] + edges.shape[1:])
         return edges, enum_lengths

class LinearChainCRF(StructDistribution):
    struct = LinearChain


class SemiMarkovCRF(StructDistribution):
    struct = SemiMarkov

class DependencyCRF(StructDistribution):
    struct = DepTree


class TreeCRF(StructDistribution):
    struct = CKY_CRF
