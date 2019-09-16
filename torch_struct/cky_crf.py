import torch
from .helpers import _Struct
from .semirings import LogSemiring
from torch.autograd import Function

A, B = 0, 1

class CKY_CRF(_Struct):
    def _dp(self, scores, lengths=None, force_grad=False):
        semiring = self.semiring
        ssize = semiring.size()
        batch, N, _, NT = scores.shape
        scores = semiring.convert(scores)
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        beta = self._make_chart(2, (batch, N, N, NT), scores, force_grad)
        span = self._make_chart(N, (batch, N, NT), scores, force_grad)
        rule_use = [
            self._make_chart(1, (batch, N - w, NT), scores, force_grad)[0]
            for w in range(N)
        ]

        # Initialize
        ns = torch.arange(N)
        rule_use[0][:] = scores[:, :, ns, ns]
        rule_use[0].requires_grad_(True)
        beta[A][:, :, ns, 0] = rule_use[0]
        beta[B][:, :, ns, N - 1] = rule_use[0]
        for w in range(1, N):
            Y = beta[A][:, :, : N - w, :w].view(ssize, batch, N - w, 1, w,  NT, 1)
            Z = beta[B][:, :, w:, N - w :].view(ssize, batch, N - w, 1, w,  1, NT)
            f = torch.arange(N-w), torch.arange(w, N)
            X = scores[:, :, f[0], f[1]].view(ssize, batch, N-w, NT)
            merge = semiring.times(Y, Z).view(ssize, batch, N - w, -1)
            rule_use[w ][:] = semiring.times(
                semiring.sum(merge, X)
            )
            span[w] = rule_use[w].view(ssize, batch, N - w, NT)
            beta[A][:, :, : N - w, w] = span[w]
            beta[B][:, :, w:N, N - w - 1] = beta[A][:, :, : N - w, w]

        final = semiring.sum(beta[A][:, :, 0, :])
        log_Z =  torch.stack(
            [final[:, b,  l - 1] for b, l in enumerate(lengths)], dim=1
        )
        return log_Z, rule_use, beta

    def _arrange_marginals(self, grads):
        semiring = self.semiring
        _, batch, N, NT = grads[0].shape
        rules = torch.zeros(batch, N, N, NT, dtype=grads[0].dtype, device=grads[0].device)

        for w, grad in enumerate(grads):
            grad = semiring.unconvert(grad)
            f = torch.arange(N - w), torch.arange(w, N)
            rules[:, f[0], f[1]] = self.semiring.unconvert(grad)
        return rules

    def enumerate(self, scores):
        semiring = self.semiring
        ssize = semiring.size()
        batch, N, _, NT = scores.shape

        def enumerate(x, start, end):
            if start + 1 == end:
                yield (scores[:, start, start, x], [(start, x)])
            else:
                for w in range(start + 1, end):
                    for y in range(NT):
                        for z in range(NT):
                            for m1, y1 in enumerate(y, start, w):
                                for m2, z1 in enumerate(z, w, end):
                                    yield (
                                        semiring.times(m1, m2, scores[:, start, end-1, x]),
                                        [(x, start, w, end)] + y1 + z1,
                                    )

        ls = []
        for nt in range(NT):
            ls += [s for s, _ in enumerate(nt, 0, N)]

        return semiring.sum(torch.stack(ls, dim=-1)), None

    @staticmethod
    def _rand():
        batch = torch.randint(2, 5, (1,))
        N = torch.randint(2, 5, (1,))
        NT = torch.randint(2, 5, (1,))
        scores = torch.rand(batch, N, N, NT)
        return scores, (batch.item(), N.item())
