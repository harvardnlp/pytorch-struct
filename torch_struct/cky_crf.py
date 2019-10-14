import torch
from .helpers import _Struct

A, B = 0, 1


class CKY_CRF(_Struct):
    def _dp(self, scores, lengths=None, force_grad=False):
        semiring = self.semiring
        ssize = semiring.size()
        batch, N, _, NT = scores.shape
        scores = semiring.convert(scores)
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        scores.requires_grad_(True)
        beta = self._make_chart(2, (batch, N, N), scores, force_grad)

        # Initialize
        reduced_scores = semiring.sum(scores)
        ns = torch.arange(N)
        rule_use = reduced_scores[:, :, ns, ns]
        beta[A][:, :, ns, 0] = rule_use
        beta[B][:, :, ns, N - 1] = rule_use

        # Run
        for w in range(1, N):
            Y = beta[A][:, :, : N - w, :w]
            Z = beta[B][:, :, w:, N - w :]
            f = torch.arange(N - w)
            X = reduced_scores[:, :, f, f + w]

            beta[A][:, :, : N - w, w] = semiring.times(
                semiring.sum(semiring.times(Y, Z)), X
            )
            beta[B][:, :, w:N, N - w - 1] = beta[A][:, :, : N - w, w]

        final = beta[A][:, :, 0]
        log_Z = torch.stack([final[:, b, l - 1] for b, l in enumerate(lengths)], dim=1)
        return log_Z, [scores], beta

    def _arrange_marginals(self, grads):
        return self.semiring.unconvert(grads[0])

    def enumerate(self, scores):
        semiring = self.semiring
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
                                        semiring.times(
                                            m1, m2, scores[:, start, end - 1, x]
                                        ),
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
