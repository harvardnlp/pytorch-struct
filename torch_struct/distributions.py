import torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property
from .linearchain import LinearChain
from .cky import CKY
from .semimarkov import SemiMarkov
from .deptree import DepTree
from .cky_crf import CKY_CRF
from .semirings import LogSemiring, MaxSemiring, EntropySemiring, MultiSampledSemiring


class StructDistribution(Distribution):
    """
    Base structured distribution class.

    Dynamic distribution for length N of structures :math:`p(z)`.

    Parameters:
        log_potentials (tensor) : batch_shape x event_shape log-potentials :math:`\phi`
        lengths (long tensor) : batch_shape integers for length masking
    """
    has_enumerate_support = True

    def __init__(self, log_potentials, lengths=None):
        batch_shape = log_potentials.shape[:1]
        event_shape = log_potentials.shape[1:]
        self.log_potentials = log_potentials
        self.lengths = lengths
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    def log_prob(self, value):
        """
        Compute log probability over values :math:`p(z)`.

        Parameters:
            value (tensor): sample_sample x batch_shape x event_shapesss
        """

        d = value.dim()
        batch_dims = range(d - len(self.event_shape))
        v = self.struct().score(
            self.log_potentials,
            value.type_as(self.log_potentials),
            batch_dims=batch_dims,
        )
        return v - self.partition

    @lazy_property
    def entropy(self):
        """
        Compute entropy for distribution :math:`H[z]`.

        Returns:
            entropy - batch_shape
        """
        return self.struct(EntropySemiring).sum(self.log_potentials, self.lengths)

    @lazy_property
    def argmax(self):
        """
        Compute an argmax for distribution :math:`\arg\max p(z)`.

        Returns:
            argmax - batch_shape x event_shape
        """
        return self.struct(MaxSemiring).marginals(self.log_potentials, self.lengths)

    @lazy_property
    def marginals(self):
        """
        Compute marginals for distribution :math:`p(z_t)`.

        Returns:
            marginals - batch_shape x event_shape
        """
        return self.struct(LogSemiring).marginals(self.log_potentials, self.lengths)

    # @constraints.dependent_property
    # def support(self):
    #     pass

    # @property
    # def param_shape(self):
    #     return self._param.size()

    @lazy_property
    def partition(self):
        "Compute the partition function."
        return self.struct(LogSemiring).sum(self.log_potentials, self.lengths)

    def sample(self, sample_shape=torch.Size()):
        """
        Compute structured samples from the distribution :math:`z \sim p(z)`.

        Parameters:
            sample_shape (int): number of samples

        Returns:
            samples - sample_shape x batch_shape x event_shape
        """
        assert len(sample_shape) == 1
        nsamples = sample_shape[0]
        samples = []
        for k in range(nsamples):
            if k % 10 == 0:
                sample = self.struct(MultiSampledSemiring).marginals(
                    self.log_potentials, lengths=self.lengths
                )
                sample = sample.detach()
            tmp_sample = MultiSampledSemiring.to_discrete(sample, (k % 10) + 1)
            samples.append(tmp_sample)
        return torch.stack(samples)

    def to_event(sequence, extra, lengths=None):
        "Convert simple representation to event."
        return self.struct.to_parts(sequence, extra, lengths=None)

    def from_event(event):
        "Convert event to simple representation."
        return self.struct.from_parts(event)

    def enumerate_support(self, expand=True):
        """
        Compute the full exponential enumeration set.

        Returns:
            (enum, enum_lengths) - tuple cardinality x batch_shape x event_shape
        """
        _, _, edges, enum_lengths = self.struct().enumerate(
            self.log_potentials, self.lengths
        )
        # if expand:
        #     edges = edges.unsqueeze(1).expand(edges.shape[:1] + self.batch_shape[:1] + edges.shape[1:])
        return edges, enum_lengths


class LinearChainCRF(StructDistribution):
    """
    Represents structured linear-chain CRFs with C classes.

    Event shape is of the form:

    Parameters:
        log_potentials (tensor) : event shape ((N-1) x C x C ) e.g.
                                  :math:`\phi(n,  z_{n+1}, z_{n})`
        lengths (long tensor) : batch_shape integers for length masking.


    Compact representation: N long tensor in [0, ..., C-1]
    """
    struct = LinearChain


class SemiMarkovCRF(StructDistribution):
    """
    Represents a semi-markov or segmental CRF with C classes of max width K

    Event shape is of the form:

    Parameters:
       log_potentials : event shape (N x K x C x C) e.g.
                        :math:`\phi(n, k, z_{n+1}, z_{n})`
       lengths (long tensor) : batch shape integers for length masking.

    Compact representation: N long tensor in [-1, 0, ..., C-1]
    """
    struct = SemiMarkov


class DependencyCRF(StructDistribution):
    """
    Represents a projective dependency CRF.

    Event shape is of the form:

    Parameters:
       log_potentials (tensor) : event shape (N x N) head, child  with
                                 arc scores with root scores on diagonal e.g.
                                 :math:`\phi(i, j)` where :math:`\phi(i, i)` is (root, i).
       lengths (long tensor) : batch shape integers for length masking.


    Compact representation: N long tensor in [0, N] (indexing is +1)
    """
    struct = DepTree


class TreeCRF(StructDistribution):
    r"""
    Represents a 0th-order span parser with NT nonterminals.

    Event shape is of the form:

    Parameters:
        log_potentials (tensor) : event_shape N x N x NT, e.g.
                                    :math:`\phi(i, j, nt)`
        lengths (long tensor) : batch shape integers for length masking.

    Compact representation:  N x N x NT long tensor (Same)
    """
    struct = CKY_CRF


class SentCFG(StructDistribution):
    """
    Represents a full generative context-free grammar with
    non-terminals NT and terminals T.

    Event shape is of the form:

    Parameters:
        log_potentials (tuple) : event tuple with event shapes
                         terms (N x T)
                         rules (NT x (NT+T) x (NT+T))
                         root  (NT)
        lengths (long tensor) : batch shape integers for length masking.

    Compact representation:  N x N x NT long tensor
    """

    struct = CKY

    def __init__(self, log_potentials, lengths=None):
        batch_shape = log_potentials[0].shape[:1]
        event_shape = log_potentials[0].shape[1:]
        self.log_potentials = log_potentials
        self.lengths = lengths
        super(StructDistribution, self).__init__(
            batch_shape=batch_shape, event_shape=event_shape
        )
