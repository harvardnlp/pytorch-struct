import torch
from .helpers import _Struct


class SemiMarkov(_Struct):
    """
    edge : b x N x K x C x C semimarkov potentials
    """

    def _check_potentials(self, edge, lengths=None):
        batch, N_1, K, C, C2 = self._get_dimension(edge)
        edge = self.semiring.convert(edge)
        N = N_1 + 1
        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(edge.device)
        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "At least one in batch must be length N"
        assert C == C2, "Transition shape doesn't match"
        return edge, batch, N, K, C, lengths

    def logpartition(self, log_potentials, lengths=None, force_grad=False):
        "Compute forward pass by linear scan"

        # Setup
        semiring = self.semiring
        ssize = semiring.size()
        log_potentials.requires_grad_(True)
        log_potentials, batch, N, K, C, lengths = self._check_potentials(
            log_potentials, lengths
        )
        log_N, bin_N = self._bin_length(N - 1)
        init = self._chart(
            (batch, bin_N, K - 1, K - 1, C, C), log_potentials, force_grad
        )

        # Init.
        semiring.one_(init.data[:, :, :, 0, 0].diagonal(0, -2, -1))

        # Length mask
        big = torch.zeros(
            ssize,
            batch,
            bin_N,
            K,
            C,
            C,
            dtype=log_potentials.dtype,
            device=log_potentials.device,
        )
        big[:, :, : N - 1] = log_potentials
        c = init[:, :, :].view(ssize, batch * bin_N, K - 1, K - 1, C, C)
        lp = big[:, :, :].view(ssize, batch * bin_N, K, C, C)
        mask = torch.arange(bin_N).view(1, bin_N).expand(batch, bin_N)
        mask = mask.to(log_potentials.device)
        mask = mask >= (lengths - 1).view(batch, 1)
        mask = mask.view(batch * bin_N, 1, 1, 1).to(lp.device)
        semiring.zero_mask_(lp.data, mask)
        semiring.zero_mask_(c.data[:, :, :, 0], (~mask))
        c[:, :, : K - 1, 0] = semiring.sum(
            torch.stack([c.data[:, :, : K - 1, 0], lp[:, :, 1:K]], dim=-1)
        )
        end = torch.min(lengths) - 1
        for k in range(1, K - 1):
            semiring.one_(
                init.data[:, :, : end - (k - 1), k - 1, k].diagonal(0, -2, -1)
            )

        K_1 = K - 1

        # Order n, n-1
        chart = (
            init.permute(0, 1, 2, 3, 5, 4, 6)
            .contiguous()
            .view(-1, batch, bin_N, K_1 * C, K_1 * C)
        )

        for n in range(1, log_N + 1):
            chart = semiring.matmul(chart[:, :, 1::2], chart[:, :, 0::2])

        final = chart.view(-1, batch, K_1, C, K_1, C)
        v = semiring.sum(semiring.sum(final[:, :, 0, :, 0, :].contiguous()))
        return v, [log_potentials]

    # def _dp_standard(self, edge, lengths=None, force_grad=False):
    #     semiring = self.semiring
    #     ssize = semiring.size()
    #     edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
    #     edge.requires_grad_(True)

    #     # Init
    #     # All paths starting at N of len K
    #     alpha = self._make_chart(1, (batch, N, K, C), edge, force_grad)[0]

    #     # All paths finishing at N with label C
    #     beta = self._make_chart(N, (batch, C), edge, force_grad)
    #     semiring.one_(beta[0].data)

    #     # Main.
    #     for n in range(1, N):
    #         alpha[:, :, n - 1] = semiring.dot(
    #             beta[n - 1].view(ssize, batch, 1, 1, C),
    #             edge[:, :, n - 1].view(ssize, batch, K, C, C),
    #         )

    #         t = max(n - K, -1)
    #         f1 = torch.arange(n - 1, t, -1)
    #         f2 = torch.arange(1, len(f1) + 1)
    #         beta[n][:] = semiring.sum(
    #             torch.stack([alpha[:, :, a, b] for a, b in zip(f1, f2)], dim=-1)
    #         )
    #     v = semiring.sum(
    #         torch.stack([beta[l - 1][:, i] for i, l in enumerate(lengths)], dim=1)
    #     )
    #     return v, [edge], beta

    @staticmethod
    def to_parts(sequence, extra, lengths=None):
        """
        Convert a sequence representation to edges

        Parameters:
            sequence : b x N  long tensors in [-1, 0, C-1]
            extra : number of states
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
