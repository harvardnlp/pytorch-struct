import torch
import math
from .semirings import LogSemiring


class Chart:
    def __init__(self, size, potentials, semiring):
        self.data = semiring.zero_(
            torch.zeros(
                *((semiring.size(),) + size),
                dtype=potentials.dtype,
                device=potentials.device
            )
        )
        self.grad = self.data.detach().clone().fill_(0.0)

    def __getitem__(self, ind):
        I = slice(None)
        return self.data[(I, I) + ind]

    def __setitem__(self, ind, new):
        I = slice(None)
        self.data[(I, I) + ind] = new


class _Struct:
    """`_Struct` is base class used to represent the graphical structure of a model.

    Subclasses should implement a `logpartition` method which computes the partition function (under the standard `_BaseSemiring`).
    Different `StructDistribution` methods will instantiate the `_Struct` subclasses
    """

    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    def logpartition(self, scores, lengths=None, force_grad=False):
        """Implement computation equivalent to the computing log partition constant logZ (if self.semiring == `_BaseSemiring`).

        Parameters:
            scores (torch.FloatTensor) : log potential scores for each factor of the model. Shape (* x batch size x *event_shape )
            lengths (torch.LongTensor) : = None, lengths of batch padded examples. Shape = ( * x batch size )
            force_grad: bool = False

        Returns:
            v (torch.Tensor) : the resulting output of the dynammic program
            logpotentials (List[torch.Tensor]): the log edge potentials of the model.
                 When `scores` is already in a log_potential format for the distribution (typical), this will be
                 [scores], as in `Alignment`, `LinearChain`, `SemiMarkov`, `CKY_CRF`.
                 An exceptional case is the `CKY` struct, which takes log potential parameters from production rules
                 for a PCFG, which are by definition independent of position in the sequence.

        """
        raise NotImplementedError()

    def score(self, potentials, parts, batch_dims=[0]):
        """Score for entire structure is product of potentials for all activated "parts"."""
        score = torch.mul(potentials, parts)  # mask potentials by activated "parts"
        batch = tuple((score.shape[b] for b in batch_dims))
        return self.semiring.prod(score.view(batch + (-1,)))

    def _bin_length(self, length):
        """Find least upper bound for lengths that is a power of 2. Used in parallel scans."""
        log_N = int(math.ceil(math.log(length, 2)))
        bin_N = int(math.pow(2, log_N))
        return log_N, bin_N

    def _get_dimension(self, edge):
        if isinstance(edge, list):
            for t in edge:
                t.requires_grad_(True)
            return edge[0].shape
        else:
            edge.requires_grad_(True)
            return edge.shape

    def _chart(self, size, potentials, force_grad):
        return self._make_chart(1, size, potentials, force_grad)[0]

    def _make_chart(self, N, size, potentials, force_grad=False):
        return [
            (
                self.semiring.zero_(
                    torch.zeros(
                        *((self.semiring.size(),) + size),
                        dtype=potentials.dtype,
                        device=potentials.device
                    )
                ).requires_grad_(force_grad and not potentials.requires_grad)
            )
            for _ in range(N)
        ]

    def sum(self, logpotentials, lengths=None, _raw=False):
        """
        Compute the (semiring) sum over all structures model.

        Parameters:
            logpotentials : generic params (see class)
            lengths: None or b long tensor mask
            _raw (bool) : return the unconverted semiring

        Returns:
            v: b tensor of total sum
        """
        v = self.logpartition(logpotentials, lengths)[0]
        if _raw:
            return v
        return self.semiring.unconvert(v)

    def marginals(self, logpotentials, lengths=None, _raw=False):
        """
        Compute the marginals of a structured model.

        Parameters:
            logpotentials : generic params (see class)
            lengths: None or b long tensor mask

        Returns:
            marginals: b x (N-1) x C x C table

        """
        with torch.autograd.enable_grad():  # in case input potentials don't have grads enabled.
            v, edges = self.logpartition(
                logpotentials, lengths=lengths, force_grad=True
            )
            if _raw:
                all_m = []
                for k in range(v.shape[0]):
                    obj = v[k].sum(dim=0)

                    marg = torch.autograd.grad(
                        obj,
                        edges,
                        create_graph=True,
                        only_inputs=True,
                        allow_unused=False,
                    )
                    all_m.append(self.semiring.unconvert(self._arrange_marginals(marg)))
                return torch.stack(all_m, dim=0)
            else:
                obj = self.semiring.unconvert(v).sum(dim=0)
                marg = torch.autograd.grad(
                    obj, edges, create_graph=True, only_inputs=True, allow_unused=False
                )
                a_m = self._arrange_marginals(marg)
                return self.semiring.unconvert(a_m)

    @staticmethod
    def to_parts(spans, extra, lengths=None):
        return spans

    @staticmethod
    def from_parts(spans):
        return spans, None

    def _arrange_marginals(self, marg):
        return marg[0]
