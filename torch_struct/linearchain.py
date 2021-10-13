r"""

A linear-chain dynamic program.

Considers parameterized functions of the form :math:`f: {\cal Y} \rightarrow \mathbb{R}`.

Combinatorial set :math:`{y_{1:N} \in \cal Y}` with each :math:`y_n \in {1, \ldots, C}`

Function factors as :math:`f(y) = \prod_{n=1}^N \phi(n, y_n, y_n{-1})`

Example use cases:

* Part-of-Speech Tagging
* Sequence Labeling
* Hidden Markov Models

"""


import torch
from .helpers import _Struct


class LinearChain(_Struct):
    """
    Represents structured linear-chain CRFs, generalizing HMMs smoothing, tagging models,
    and anything with chain-like dynamics.
    """

    def _check_potentials(self, edge, lengths=None):
        batch, N_1, C, C2 = self._get_dimension(edge)
        edge = self.semiring.convert(edge)
        N = N_1 + 1

        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(edge.device)
            # pass
        else:
            assert max(lengths) <= N, "Length longer than edge scores"
            assert max(lengths) == N, "One length must be at least N"
        assert C == C2, "Transition shape doesn't match"
        return edge, batch, N, C, lengths

    def logpartition(self, log_potentials, lengths=None, force_grad=False):
        "Compute forward pass by linear scan"
        # Setup
        semiring = self.semiring
        ssize = semiring.size()
        log_potentials, batch, N, C, lengths = self._check_potentials(
            log_potentials, lengths
        )
        log_N, bin_N = self._bin_length(N - 1)
        chart = self._chart((batch, bin_N, C, C), log_potentials, force_grad)

        # Init
        init = torch.zeros_like(chart).bool()
        init.diagonal(0, 3, 4).fill_(True)
        chart = semiring.fill(chart, init, semiring.one)

        # Length mask
        big = torch.zeros(
            ssize,
            batch,
            bin_N,
            C,
            C,
            dtype=log_potentials.dtype,
            device=log_potentials.device,
        )
        big[:, :, : N - 1] = log_potentials
        c = chart[:, :, :].view(ssize, batch * bin_N, C, C)
        lp = big[:, :, :].view(ssize, batch * bin_N, C, C)
        mask = torch.arange(bin_N).view(1, bin_N).expand(batch, bin_N).type_as(c)
        mask = mask >= (lengths - 1).view(batch, 1)
        mask = mask.view(batch * bin_N, 1, 1).to(lp.device)
        lp.data[:] = semiring.fill(lp.data, mask, semiring.zero)
        c.data[:] = semiring.fill(c.data, ~mask, semiring.zero)

        c[:] = semiring.sum(torch.stack([c.data, lp], dim=-1))

        # Scan
        for n in range(1, log_N + 1):
            chart = semiring.matmul(chart[:, :, 1::2], chart[:, :, 0::2])
        v = semiring.sum(semiring.sum(chart[:, :, 0].contiguous()))
        return v, [log_potentials]

    @staticmethod
    def to_parts(sequence, extra, lengths=None):
        """
        Convert a sequence representation to edges

        Parameters:
            sequence : b x N long tensor in [0, C-1]
            extra : number of states
            lengths: b long tensor of N values

        Returns:
            edge : b x (N-1) x C x C markov indicators
                        (t x z_t x z_{t-1})
        """
        C = extra
        batch, N = sequence.shape
        labels = torch.zeros(batch, N - 1, C, C).long()
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        for n in range(1, N):
            labels[torch.arange(batch), n - 1, sequence[:, n], sequence[:, n - 1]] = 1
        for b in range(batch):
            labels[b, lengths[b] - 1 :, :, :] = 0
        return labels

    @staticmethod
    def from_parts(edge):
        """
        Convert edges to sequence representation.

        Parameters:
            edge : b x (N-1) x C x C markov indicators
                        (t x z_t x z_{t-1})

        Returns:
            sequence : b x N long tensor in [0, C-1]
        """
        batch, N_1, C, _ = edge.shape
        N = N_1 + 1
        labels = torch.zeros(batch, N).long()
        on = edge.nonzero()
        for i in range(on.shape[0]):
            if on[i][1] == 0:
                labels[on[i][0], on[i][1]] = on[i][3]
            labels[on[i][0], on[i][1] + 1] = on[i][2]
        return labels, C

    # Adapters
    @staticmethod
    def hmm(transition, emission, init, observations):
        """
        Convert HMM log-probs to a linear chain.

        Parameters:
            transition: C X C
            emission: V x C
            init: C
            observations: b x N between [0, V-1]

        Returns:
            edges: b x (N-1) x C x C
        """
        V, C = emission.shape
        batch, N = observations.shape
        scores = torch.zeros(batch, N - 1, C, C).type_as(emission)
        scores[:, :, :, :] += transition.view(1, 1, C, C)
        scores[:, 0, :, :] += init.view(1, 1, C)
        obs = emission[observations.view(batch * N), :]
        scores[:, :, :, :] += obs.view(batch, N, C, 1)[:, 1:]
        scores[:, 0, :, :] += obs.view(batch, N, 1, C)[:, 0]
        return scores

    @staticmethod
    def _rand(min_n=2):
        b = torch.randint(2, 4, (1,))
        N = torch.randint(min_n, 4, (1,))
        C = torch.randint(2, 4, (1,))
        return torch.rand(b, N, C, C), (b.item(), (N + 1).item())
