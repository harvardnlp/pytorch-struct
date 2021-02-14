import torch
from .helpers import _Struct
import math

try:
    import genbmm
except ImportError:
    pass

from .semirings import LogSemiring
from .semirings.fast_semirings import broadcast

Down, Mid, Up = 0, 1, 2
Open, Close = 0, 1


class Alignment(_Struct):
    def __init__(
        self, semiring=LogSemiring, sparse_rounds=3, max_gap=None, local=False
    ):
        self.semiring = semiring
        self.sparse_rounds = sparse_rounds
        self.local = local
        self.max_gap = max_gap

    def _check_potentials(self, edge, lengths=None):
        batch, N_1, M_1, x = edge.shape
        assert x == 3
        if self.local:
            assert (edge[..., 0] <= 0).all(), "skips must be negative"
            assert (edge[..., 1] >= 0).all(), "alignment must be positive"
            assert (edge[..., 2] <= 0).all(), "skips must be negative"

        edge = self.semiring.convert(edge)

        N = N_1
        M = M_1

        assert M >= N

        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(edge.device)

        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "One length must be at least N"
        return edge, batch, N, M, lengths

    def logpartition(self, log_potentials, lengths=None, force_grad=False, cache=True):
        return self._dp_scan(log_potentials, lengths, force_grad)

    def _dp_scan(self, log_potentials, lengths=None, force_grad=False):
        "Compute forward pass by linear scan"
        # Setup
        semiring = self.semiring
        log_potentials.requires_grad_(True)
        ssize = semiring.size()
        log_potentials, batch, N, M, lengths = self._check_potentials(
            log_potentials, lengths
        )

        # N is the longer (time) dimension.
        steps = M + N
        log_N = int(math.ceil(math.log(steps, 2)))
        bin_N = int(math.pow(2, log_N))
        LOC = 2 if self.local else 1

        # Create a chart N, N, back
        charta = [None, None]

        # offset = 1, left_pos = bin_N
        charta[0] = self._make_chart(
            1, (batch, bin_N, 1, bin_N, LOC, LOC, 3), log_potentials, force_grad
        )[0]
        charta[1] = self._make_chart(
            1, (batch, bin_N // 2, 3, bin_N, LOC, LOC, 3), log_potentials, force_grad
        )[0]

        # Init
        # This part is complicated. Rotate the scores by 45% and
        # then compress one.
        grid_x = torch.arange(N).view(N, 1).expand(N, M)
        grid_y = torch.arange(M).view(1, M).expand(N, M)
        rot_x = grid_x + grid_y
        rot_y = grid_y - grid_x + N - 1

        ind = torch.arange(bin_N)
        ind_M = ind
        ind_U = torch.arange(1, bin_N)
        ind_D = torch.arange(bin_N - 1)
        for b in range(lengths.shape[0]):
            # Fill base chart with values.
            l = lengths[b]
            charta[0][:, b, rot_x[:l], 0, rot_y[:l], :, :, :] = log_potentials[
                :, b, :l, :, None, None
            ]

            # Create finalizing paths.
            point = (l + M) // 2

            charta[1][:, b, point:, 1, ind, :, :, Mid] = semiring.one_(
                charta[1][:, b, point:, 1, ind, :, :, Mid]
            )

        for b in range(lengths.shape[0]):
            point = (lengths[b] + M) // 2
            lim = point * 2

            left_ = charta[0][:, b, 0:lim:2, 0]
            right = charta[0][:, b, 1:lim:2, 0]

            charta[1][:, b, :point, 1, ind_M] = torch.stack(
                [
                    left_[..., Down],
                    semiring.plus(left_[..., Mid], right[..., Mid]),
                    left_[..., Up],
                ],
                dim=-1,
            )

            y = torch.stack([ind_D, ind_U], dim=0)
            z = y.clone()
            z[0, :] = 2
            z[1, :] = 0

            charta[1][:, b, :point, z, y, :, :, :] = torch.stack(
                [
                    semiring.times(
                        left_[:, :, ind_D, Open : Open + 1 :, :],
                        right[:, :, ind_U, :, Open : Open + 1, Down : Down + 1],
                    ),
                    semiring.times(
                        left_[:, :, ind_U, Open : Open + 1, :, :],
                        right[:, :, ind_D, :, Open : Open + 1, Up : Up + 1],
                    ),
                ],
                dim=2,
            )

        chart = charta[1][..., :, :, :].permute(0, 1, 2, 5, 6, 7, 4, 3)

        # Scan
        def merge(x):
            inner = x.shape[-1]
            width = (inner - 1) // 2
            left = x[:, :, 0::2, Open, :].view(
                ssize, batch, -1, 1, LOC, 3, bin_N, inner
            )
            right = x[:, :, 1::2, :, Open].view(
                ssize, batch, -1, LOC, 1, 1, 3, bin_N, inner
            )

            st = []
            for op in (Mid, Up, Down):
                leftb, rightb, _ = broadcast(left, right[..., op, :, :])
                leftb = genbmm.BandedMatrix(leftb, width, width, semiring.zero)
                rightb = genbmm.BandedMatrix(rightb, width, width, semiring.zero)
                leftb = leftb.transpose().col_shift(op - 1).transpose()
                v = semiring.matmul(rightb, leftb).band_pad(1).band_shift(op - 1)
                v = v.data.view(ssize, batch, -1, LOC, LOC, 3, bin_N, v.data.shape[-1])
                st.append(v)

            if self.local:

                def pad(v):
                    s = list(v.shape)
                    s[-1] = inner // 2 + 1
                    pads = torch.zeros(*s, device=v.device, dtype=v.dtype).fill_(
                        semiring.zero
                    )
                    return torch.cat([pads, v, pads], -1)

                left_ = x[:, :, 0::2, Close, None]
                left_ = pad(left)
                right = x[:, :, 1::2, :, Close, None]
                right = pad(right)
                st.append(torch.cat([semiring.zero_(left_.clone()), left_], dim=3))
                st.append(torch.cat([semiring.zero_(right.clone()), right], dim=4))
            return semiring.sum(torch.stack(st, dim=-1))

        for n in range(2, log_N + 1):
            chart = merge(chart)

            center = int((chart.shape[-1] - 1) // 2)
            if center > (bin_N / 2):
                chart = chart[..., center - (bin_N // 2) : center + (bin_N // 2) + 1]
            elif self.max_gap is not None and center > self.max_gap:
                chart = chart[..., center - self.max_gap : center + self.max_gap + 1]

        if self.local:
            v = semiring.sum(semiring.sum(chart[..., 0, Close, Close, Mid, :, :]))
        else:
            v = chart[
                ..., 0, Open, Open, Mid, N - 1, M - N + ((chart.shape[-1] - 1) // 2)
            ]
        return v, [log_potentials]
