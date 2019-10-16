import torch
from .helpers import _Struct
import math


class LinearChain(_Struct):
    """
    Represents structured linear-chain CRFs, generalizing HMMs smoothing, tagging models,
    and anything with chain-like dynamics.
    """

    def _check_potentials(self, edge, lengths=None):
        batch, N_1, C, C2 = edge.shape
        edge = self.semiring.convert(edge)

        N = N_1 + 1

        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "One length must be at least N"
        assert C == C2, "Transition shape doesn't match"
        return edge, batch, N, C, lengths

    def _dp(self, log_potentials, lengths=None, force_grad=False):
        return self._dp_scan(log_potentials, lengths, force_grad)

    def _dp_scan(self, log_potentials, lengths=None, force_grad=False):
        "Compute forward pass by linear scan"
        semiring = self.semiring
        log_potentials.requires_grad_(True)
        ssize = semiring.size()
        log_potentials, batch, N, C, lengths = self._check_potentials(
            log_potentials, lengths
        )
        log_N = int(math.ceil(math.log(N - 1, 2)))
        bin_N = int(math.pow(2, log_N))

        # setup scan
        def left(x, size):
            return x[:, :, 0 : size * 2 : 2]

        def right(x, size):
            return x[:, :, 1 : size * 2 : 2]

        def root(x):
            return x[:, :, 0]

        def merge(x, y, size):
            return semiring.dot(
                x.transpose(3, 4).view(ssize, batch, size, 1, C, C),
                y.view(ssize, batch, size, C, 1, C),
            )

        chart = self._make_chart(
            log_N + 1, (batch, bin_N, C, C), log_potentials, force_grad
        )

        # Init
        for b in range(lengths.shape[0]):
            end = lengths[b] - 1
            semiring.zero_(chart[0][:, b, end:])
            chart[0][:, b, end:, torch.arange(C), torch.arange(C)] = semiring.one_(
                chart[0][:, b, end:, torch.arange(C), torch.arange(C)]
            )

        for b in range(lengths.shape[0]):
            end = lengths[b] - 1
            chart[0][:, b, :end] = log_potentials[:, b, :end]

        # Scan
        size = bin_N
        for n in range(1, log_N + 1):
            size = int(size / 2)
            chart[n][:, :, :size] = merge(
                left(chart[n - 1], size), right(chart[n - 1], size), size
            )
        print(root(chart[-1][:]))
        v = semiring.sum(semiring.sum(root(chart[-1][:])))

        return v, [log_potentials], None

    def _dp_standard(self, edge, lengths=None, force_grad=False):
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, C, lengths = self._check_potentials(edge, lengths)

        alpha = self._make_chart(N, (batch, C), edge, force_grad)
        edge_store = self._make_chart(N - 1, (batch, C, C), edge, force_grad)

        semiring.one_(alpha[0].data)

        for n in range(1, N):
            edge_store[n - 1][:] = semiring.times(
                alpha[n - 1].view(ssize, batch, 1, C),
                edge[:, :, n - 1].view(ssize, batch, C, C),
            )
            alpha[n][:] = semiring.sum(edge_store[n - 1])

        for n in range(1, N):
            edge_store[n - 1][:] = semiring.times(
                alpha[n - 1].view(ssize, batch, 1, C),
                edge[:, :, n - 1].view(ssize, batch, C, C),
            )
            alpha[n][:] = semiring.sum(edge_store[n - 1])

        ret = [alpha[lengths[i] - 1][:, i] for i in range(batch)]
        ret = torch.stack(ret, dim=1)
        v = semiring.sum(ret)
        return v, edge_store, alpha

    @staticmethod
    def to_parts(sequence, extra, lengths=None):
        """
        Convert a sequence representation to edges

        Parameters:
            sequence : b x N long tensor in [0, C-1]
            C : number of states
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
        Convert HMM to a linear chain.

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
        scores = torch.ones(batch, N - 1, C, C).type_as(emission)
        scores[:, :, :, :] *= transition.view(1, 1, C, C)
        scores[:, 0, :, :] *= init.view(1, 1, C)
        obs = emission[observations.view(batch * N), :]
        scores[:, :, :, :] *= obs.view(batch, N, C, 1)[:, 1:]
        scores[:, 0, :, :] *= obs.view(batch, N, 1, C)[:, 0]
        return scores

    @staticmethod
    def _rand(min_n=2):
        b = torch.randint(2, 4, (1,))
        N = torch.randint(min_n, 4, (1,))
        C = torch.randint(2, 4, (1,))
        return torch.rand(b, N, C, C), (b.item(), (N + 1).item())

    ### Tests

    def enumerate(self, edge, lengths=None):
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, C, lengths = self._check_potentials(edge, lengths)
        chains = [[([c], semiring.one_(torch.zeros(ssize, batch))) for c in range(C)]]

        enum_lengths = torch.LongTensor(lengths.shape)
        for n in range(1, N):
            new_chains = []
            for chain, score in chains[-1]:
                for c in range(C):
                    new_chains.append(
                        (
                            chain + [c],
                            semiring.mul(score, edge[:, :, n - 1, c, chain[-1]]),
                        )
                    )
            chains.append(new_chains)

            for b in range(lengths.shape[0]):
                if lengths[b] == n + 1:
                    enum_lengths[b] = len(new_chains)

        edges = self.to_parts(
            torch.stack([torch.tensor(c) for (c, _) in chains[-1]]), C
        )
        # Sum out non-batch
        a = torch.einsum("ancd,sbncd->sbancd", edges.float(), edge)
        a = semiring.prod(a.view(*a.shape[:3] + (-1,)), dim=3)
        a = semiring.sum(a, dim=2)
        ret = semiring.sum(torch.stack([s for (_, s) in chains[-1]], dim=1), dim=1)
        assert torch.isclose(a, ret).all(), "%s %s" % (a, ret)

        edges = torch.zeros(len(chains[-1]), batch, N - 1, C, C)
        for b in range(lengths.shape[0]):
            edges[: enum_lengths[b], b, : lengths[b] - 1] = self.to_parts(
                torch.stack([torch.tensor(c) for (c, _) in chains[lengths[b] - 1]]), C
            )

        return (
            semiring.unconvert(ret),
            [s for (_, s) in chains[-1]],
            edges,
            enum_lengths,
        )

        # def parent(x, size):
        # return x[:, :, :size]

        # # Scan downward
        # print("a",  v)
        # semiring.zero_(root(chart2[-1][:]))
        # chart2[-1][:, :, 0, torch.arange(C), torch.arange(C)] = semiring.one_(
        #     chart2[-1][:, :, 0, torch.arange(C), torch.arange(C)]
        # )

        # size = 1
        # for n in range(log_N - 1, -1, -1):
        #     left(chart2[n], size)[:] = parent(chart2[n + 1], size)
        #     right(chart2[n], size)[:] = merge(
        #         parent(chart2[n + 1], size), left(chart[n], size), size
        #     )
        #     size = size * 2

        # final = merge(chart2[0], chart[0], size)
        # ret = [final[:, i, lengths[i] - 2] for i in range(batch)]
        # ret = torch.stack(ret, dim=1)
        # v2 = semiring.sum(semiring.sum(ret))
        # print("b", v2)

    # def _dp_backward(self, edge, lengths, alpha_in, v=None):
    #     semiring = self.semiring
    #     batch, N, C, lengths = self._check_potentials(edge, lengths)

    #     alpha = self._make_chart(N, (batch, C), edge, force_grad=False)
    #     edge_store = self._make_chart(N - 1, (batch, C, C), edge, force_grad=False)

    #     for n in range(N - 1, 0, -1):
    #         for b, l in enumerate(lengths):
    #             alpha[l - 1][b].data.fill_(semiring.one())

    #         edge_store[n - 1][:] = semiring.times(
    #             alpha[n].view(batch, C, 1), edge[:, n - 1]
    #         )
    #         alpha[n - 1][:] = semiring.sum(edge_store[n - 1], dim=-2)
    #     v = semiring.sum(
    #         torch.stack([alpha[0][i] for i, l in enumerate(lengths)]), dim=-1
    #     )
    #     edge_marginals = self._make_chart(
    #         1, (batch, N - 1, C, C), edge, force_grad=False
    #     )[0]

    #     for n in range(N - 1):
    #         edge_marginals[:, n] = semiring.div_exp(
    #             semiring.times(
    #                 alpha_in[n].view(batch, 1, C),
    #                 edge[:, n],
    #                 alpha[n + 1].view(batch, C, 1),
    #             ),
    #             v.view(batch, 1, 1),
    #         )

    #     return edge_marginals
