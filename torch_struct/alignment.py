import torch
from .helpers import _Struct
from .semirings import LogSemiring
import math

def pad_conv(x, k, dim, sr, extra_b=0, extra_t=0):
    return pad(x, (k - 1) // 2 + extra_b, (k-1)//2 + extra_t, dim, sr).unfold(dim, k, 1)

def pad(x, n_bot, n_top, dim, sr):
    shape = list(x.shape)
    shape[dim] = n_bot
    padb = sr.zero_(torch.zeros(shape, dtype=x.dtype, device=x.device))
    shape[dim] = n_top
    padt = sr.zero_(torch.zeros(shape, dtype=x.dtype, device=x.device))

    return torch.cat([padb, x, padt], dim=dim)

def demote(x, index):
    total = x.dim()
    order = tuple(range(index)) + tuple(range(index + 1, total)) + (index, )
    return x.permute(order)

class Alignment(_Struct):
    def __init__(self, semiring=LogSemiring, local=False, max_gap=None):
        self.semiring = semiring
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
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "One length must be at least N"
        return edge, batch, N, M, lengths

    def _dp(self, log_potentials, lengths=None, force_grad=False):
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
        assert self.max_gap is None or self.max_gap > abs(N - M)

        steps = N + M
        log_MN = int(math.ceil(math.log(steps, 2)))
        bin_MN = int(math.pow(2, log_MN))

        Down, Mid, Up = 0, 1, 2
        Open, Close = 0, 1


        # Grid
        grid_x = torch.arange(N).view(N, 1).expand(N, M)
        grid_y = torch.arange(M).view(1, M).expand(N, M)
        rot_x = grid_x + grid_y
        rot_y = grid_y - grid_x + N

        # Helpers
        ind = torch.arange(bin_MN)
        ind_M = ind
        ind_U = torch.arange(1, bin_MN)
        ind_D = torch.arange(bin_MN - 1)

        # Create a chart N, N, back
        # chart = [self._make_chart(
        #     1,
        #     (batch, bin_MN // pow(2, i), bin_MN, bin_MN, 2, 2, 3),
        #     log_potentials,
        #     force_grad,
        # )[0] for i in range(log_MN + 1)]


        charta = [self._make_chart(
            1,
            (batch, bin_MN // pow(2, i), 2 * bin_MN // pow(2, log_MN - i)-1, bin_MN, 2, 2, 3),
            log_potentials,
            force_grad,
        )[0] for i in range(log_MN + 1)]

        chartb = [self._make_chart(
            1,
            (batch, bin_MN // pow(2, i), bin_MN, 2* bin_MN // pow(2, log_MN- i) -1, 2, 2, 3),
            log_potentials,
            force_grad,
        )[0] for i in range(log_MN + 1)]


        def reflect(x, size):
            ex = x.shape[3]
            f, r = torch.arange(ex), torch.arange(ex-1, -1, -1)
            sp = pad_conv(x, ex, 4, semiring)
            # print(sp.shape)
            # print(bin_MN)
            # print((ssize, batch, size, ex, bin_MN,  2, 2, 3, ex))
            sp.view(ssize, batch, size, ex, bin_MN,  2, 2, 3, ex)
            sp = sp[:, :, :, r, :, :, :, :, f].permute(1,2,3,4,0,5,6,7) \
                                              .view(ssize, batch, size, bin_MN, ex, 2, 2, 3)
            return sp

        # Init
        # This part is complicated. Rotate the scores by 45% and
        # then compress one.

        for b in range(lengths.shape[0]):
            end = lengths[b]
            point = (end + M) // 2
            lim = point * 2
            # OLD
            # chart[0][:, b, rot_x[:lim], rot_y[:lim], rot_y[:lim], :, :, :] = (
            #     log_potentials[:, b, :lim].unsqueeze(-2).unsqueeze(-2)
            # )
            # OLD
            charta[0][:, b, rot_x[:lim], 0, rot_y[:lim], :, :, :] = (
                log_potentials[:, b, :lim].unsqueeze(-2).unsqueeze(-2)
            )
            chartb[0][:, b, rot_x[:lim], rot_y[:lim], 0, :, :, :] = (
                log_potentials[:, b, :lim].unsqueeze(-2).unsqueeze(-2)
            )

            # OLD
            # chart[1][:, b, point:, ind, ind, :, :, Mid] = semiring.one_(
            #     chart[1][:, b, point:, ind, ind, :, :, Mid]
            # )
            # OLD

            charta[1][:, b, point:, 1, ind, :, :, Mid] = semiring.one_(
                charta[1][:, b, point:, 1, ind, :, :, Mid]
            )

        for b in range(lengths.shape[0]):
            end = lengths[b]
            point = (end + M) // 2
            lim = point * 2
            # left_ = chart[0][:, b, 0:lim:2]
            # right = chart[0][:, b, 1:lim:2]

            left2_ = charta[0][:, b, 0:lim:2]
            right2 = chartb[0][:, b, 1:lim:2]

            # chart[1][:, b, :point, ind_M, ind_M, :, :, :] = torch.stack(
            #     [
            #         left_[:, :, ind_M, ind_M, :, :, Down],
            #         semiring.plus(
            #             left_[:, :, ind_M, ind_M, :, :, Mid],
            #             right[:, :, ind_M, ind_M, :, :, Mid]),
            #         left_[:, :, ind_M, ind_M, :, :, Up],
            #     ],
            #     dim=-1,
            # )

            charta[1][:, b, :point, 1, ind_M, :, :, :] = torch.stack(
                [
                    left2_[:, :, 0, ind_M, :, :, Down],
                    semiring.plus(
                        left2_[:, :, 0, ind_M, :, :, Mid],
                        right2[:, :, ind_M, 0, :, :, Mid]),
                    left2_[:, :, 0, ind_M, :, :, Up],
                ],
                dim=-1,
            )

            x = torch.stack([ind_U,
                             ind_D], dim=0)
            y = torch.stack([ind_D,
                             ind_U], dim=0)
            z = y.clone()
            z[0, :] = 2
            z[1, :] = 0

            z2 = y.clone()
            z2[0, :] = 0
            z2[1, :] = 2

            tmp = torch.stack(
                [
                    semiring.times(
                        left2_[:, :, 0, ind_D, Open : Open+1 :, :],
                        right2[:, :, ind_U, 0, :, Open : Open + 1, Down : Down + 1]
                    ),
                    semiring.times(
                        left2_[:, :, 0, ind_U, Open : Open + 1, :, :],
                        right2[:, :, ind_D, 0, :, Open : Open + 1, Up : Up + 1],
                    ),
                ],
                dim=2,
            )
            charta[1][:, b, :point, z, y, :, :, :] = tmp



        charta[1] = charta[1][:, :, :, :3]
        chartb[1] = reflect(charta[1], bin_MN // 2)

        # Scan
        def merge2(xa, xb, size, rsize):
            nrsize = (rsize-1)*2+3
            rsize += 2
            print(nrsize, rsize)
            st = []
            left = (
                pad_conv(demote(xa[:, :, 0 : size * 2 : 2, :], 3), nrsize, 7, semiring, 2, 2)
                .transpose(-1, -2)
                .view(ssize, batch, size, bin_MN, 1, 2, 2, 3, nrsize, rsize+2)
            )

            right = (
                pad(pad_conv(demote(xb[:, :, 1 : size * 2 : 2, :, :], 4), nrsize, 3, semiring), 1, 1, -2, semiring)
                .transpose(-1, -2)
                .view(ssize, batch, size, bin_MN, 2, 1, 2, 1, 3, nrsize, rsize)
            )

            for op in (Up, Down, Mid):
                top, bot = rsize + 1, 1
                if op == Up:
                    top, bot = rsize +2 , 2
                if op == Down:
                    top, bot = rsize, 0

                combine = semiring.dot(
                    left[:, :, :, :, :, Open, :, :, :, bot:top],
                    right[:, :, :, :, :, Open, :, :, op, :, :]
                )
                combine = combine.view(ssize, batch, size, bin_MN, 2, 2, 3, nrsize) \
                                 .permute(0, 1, 2, 7, 3, 4, 5, 6)
                st.append(combine)

            if self.local:
                left_ = pad(xa[:, :, 0 :: 2, :, :, Close, :, :], rsize//2, rsize//2, 3, semiring)
                right = pad(xa[:, :, 1 :: 2, :, :, :, Close, :], rsize//2, rsize//2, 3, semiring)
                st.append(torch.stack([semiring.zero_(left_.clone()), left_], dim=-3))
                st.append(torch.stack([semiring.zero_(right.clone()), right], dim=-2))
            st = torch.stack(st, dim=-1)
            return semiring.sum(st)


        size = bin_MN // 2
        rsize = 2
        for n in range(2, log_MN + 1):
            print(n)
            size = int(size / 2)
            rsize *= 2
            q = merge2(charta[n - 1], chartb[n - 1], size, charta[n - 1].shape[3])
            charta[n] = q
            gap = charta[n].shape[3]
            if self.max_gap is not None and (gap - 1) // 2 > self.max_gap:

                reduced = (gap - 1) // 2 - self.max_gap
                charta[n] = charta[n][:, :, :, reduced:-reduced]
                chartb[n] = reflect(charta[n], size)
            else:
                chartb[n] = reflect(q, size)


            # Old
            # chart[n][:] = merge(chart[n - 1], size)
            # Old

        if self.local:
            v = semiring.sum(semiring.sum(charta[-1][:, :, 0, :, :, Close, Close, Mid]))
        else:
            # v = chart[-1][:, :, 0, M, N, Open, Open, Mid]
            v = charta[-1][:, :, 0,  M-N + (charta[-1].shape[3] //2),
                           N, Open, Open, Mid]
        return v, [log_potentials], None

    @staticmethod
    def _rand(min_n=2):
        b = torch.randint(2, 3, (1,))
        N = torch.randint(min_n, 6, (1,))
        M = torch.randint(min_n, 6, (1,))
        return torch.rand(b, N, M, 3), (b.item(), (N).item())

    def enumerate(self, edge, lengths=None):
        semiring = self.semiring
        edge, batch, N, M, lengths = self._check_potentials(edge, lengths)
        d = {}
        d[0, 0] = [([(0, 0, 1)], edge[:, :, 0, 0, 1])]
        # enum_lengths = torch.LongTensor(lengths.shape)
        if self.local:
            for i in range(N):
                for j in range(M):
                    d.setdefault((i, j), [])
                    d[i, j].append(([(i, j, 1)], edge[:, :, i, j, 1]))

        for i in range(N):
            for j in range(M):
                d.setdefault((i + 1, j + 1), [])
                d.setdefault((i, j + 1), [])
                d.setdefault((i + 1, j), [])
                for chain, score in d[i, j]:
                    if i + 1 < N and j + 1 < M:
                        d[i + 1, j + 1].append(
                            (
                                chain + [(i + 1, j + 1, 1)],
                                semiring.mul(score, edge[:, :, i + 1, j + 1, 1]),
                            )
                        )
                    if i + 1 < N:

                        d[i + 1, j].append(
                            (
                                chain + [(i + 1, j, 2)],
                                semiring.mul(score, edge[:, :, i + 1, j, 2]),
                            )
                        )
                    if j + 1 < M:
                        d[i, j + 1].append(
                            (
                                chain + [(i, j + 1, 0)],
                                semiring.mul(score, edge[:, :, i, j + 1, 0]),
                            )
                        )
        if self.local:
            positions = [x[0] for i in range(N) for j in range(M) for x in d[i, j]]
            all_val = torch.stack(
                [x[1] for i in range(N) for j in range(M) for x in d[i, j]], dim=-1
            )
            _, ind = all_val.max(dim=-1)
            print(positions[ind[0, 0]])
        else:
            all_val = torch.stack([x[1] for x in d[N - 1, M - 1]], dim=-1)

        return semiring.unconvert(semiring.sum(all_val)), None
