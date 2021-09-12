import torch
import math
from .semirings import LogSemiring


class Chart:
    def __init__(self, size, potentials, semiring):
        c = torch.zeros(
            *((semiring.size(),) + size),
            dtype=potentials.dtype,
            device=potentials.device
        )
        c[:] = semiring.zero.view((semiring.size(),) + len(size) * (1,))

        self.data = c
        self.grad = self.data.detach().clone().fill_(0.0)

    def __getitem__(self, ind):
        I = slice(None)
        return self.data[(I, I) + ind]

    def __setitem__(self, ind, new):
        I = slice(None)
        self.data[(I, I) + ind] = new


class _Struct:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    def score(self, potentials, parts, batch_dims=[0]):
        score = torch.mul(potentials, parts)
        batch = tuple((score.shape[b] for b in batch_dims))
        return self.semiring.prod(score.view(batch + (-1,)))

    def _bin_length(self, length):
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
        chart = []
        for _ in range(N):
            c = torch.zeros(
                *((self.semiring.size(),) + size),
                dtype=potentials.dtype,
                device=potentials.device
            )
            c[:] = self.semiring.zero.view((self.semiring.size(),) + len(size) * (1,))
            c.requires_grad_(force_grad and not potentials.requires_grad)
            chart.append(c)
        return chart

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
        v, edges = self.logpartition(logpotentials, lengths=lengths, force_grad=True)
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
