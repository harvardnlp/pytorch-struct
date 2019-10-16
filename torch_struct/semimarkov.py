import torch
from .helpers import _Struct


class SemiMarkov(_Struct):
    """
    edge : b x N x K x C x C semimarkov potentials
    """

    def _check_potentials(self, edge, lengths=None):
        batch, N_1, K, C, C2 = edge.shape
        edge = self.semiring.convert(edge)
        N = N_1 + 1
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "At least one in batch must be length N"
        assert C == C2, "Transition shape doesn't match"
        return edge, batch, N, K, C, lengths

    def _dp(self, edge, lengths=None, force_grad=False):
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
        edge.requires_grad_(True)

        # Init
        # All paths starting at N of len K
        alpha = self._make_chart(1, (batch, N, K, C), edge, force_grad)[0]

        # All paths finishing at N with label C
        beta = self._make_chart(N, (batch, C), edge, force_grad)
        semiring.one_(beta[0].data)

        # Main.
        for n in range(1, N):
            alpha[:, :, n - 1] = semiring.dot(
                beta[n - 1].view(ssize, batch, 1, 1, C),
                edge[:, :, n - 1].view(ssize, batch, K, C, C),
            )

            t = max(n - K, -1)
            f1 = torch.arange(n - 1, t, -1)
            f2 = torch.arange(1, len(f1) + 1)
            beta[n][:] = semiring.sum(
                torch.stack([alpha[:, :, a, b] for a, b in zip(f1, f2)], dim=-1)
            )
        v = semiring.sum(
            torch.stack([beta[l - 1][:, i] for i, l in enumerate(lengths)], dim=1)
        )
        return v, [edge], beta

    @staticmethod
    def _rand():
        b = torch.randint(2, 4, (1,))
        N = torch.randint(2, 4, (1,))
        K = torch.randint(2, 4, (1,))
        C = torch.randint(2, 4, (1,))
        return torch.rand(b, N, K, C, C), (b.item(), (N + 1).item())

    @staticmethod
    def to_parts(sequence, extra, lengths=None):
        """
        Convert a sequence representation to edges

        Parameters:
            sequence : b x N  long tensors in [-1, 0, C-1]
            C : number of states
            lengths: b long tensor of N values
        Returns:
            edge : b x (N-1) x K x C x C semimarkov potentials
                        (t x z_t x z_{t-1})
        """
        C, K = extra
        batch, N = sequence.shape
        labels = torch.zeros(batch, N - 1, K, C, C).long()
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        for b in range(batch):
            last = None
            c = None
            for n in range(0, N):
                if sequence[b, n] == -1:
                    assert n != 0
                    continue
                else:
                    new_c = sequence[b, n]
                    if n != 0:
                        labels[b, last, n - last, new_c, c] = 1
                    last = n
                    c = new_c
        return labels

    @staticmethod
    def from_parts(edge):
        """
        Convert a edges to a sequence representation.

        Parameters:
            edge : b x (N-1) x K x C x C semimarkov potentials
                    (t x z_t x z_{t-1})
        Returns:
            sequence : b x N  long tensors in [-1, 0, C-1]

        """
        batch, N_1, K, C, _ = edge.shape
        N = N_1 + 1
        labels = torch.zeros(batch, N).long().fill_(-1)
        on = edge.nonzero()
        for i in range(on.shape[0]):
            if on[i][1] == 0:
                labels[on[i][0], on[i][1]] = on[i][4]
            labels[on[i][0], on[i][1] + on[i][2]] = on[i][3]
        # print(edge.nonzero(), labels)
        return labels, (C, K)

    # Tests
    def enumerate(self, edge):
        semiring = self.semiring
        ssize = semiring.size()
        batch, N, K, C, _ = edge.shape
        edge = semiring.convert(edge)
        chains = {}
        chains[0] = [
            ([(c, 0)], semiring.one_(torch.zeros(ssize, batch))) for c in range(C)
        ]

        for n in range(1, N + 1):
            chains[n] = []
            for k in range(1, K):
                if n - k not in chains:
                    continue
                for chain, score in chains[n - k]:
                    for c in range(C):
                        chains[n].append(
                            (
                                chain + [(c, k)],
                                semiring.mul(
                                    score, edge[:, :, n - k, k, c, chain[-1][0]]
                                ),
                            )
                        )
        ls = [s for (_, s) in chains[N]]
        return semiring.unconvert(semiring.sum(torch.stack(ls, dim=1), dim=1)), ls
