import torch
from .helpers import _Struct
from .semirings import LogSemiring
from torch.autograd import Function

A, B = 0, 1

class CKY_CRF(_Struct):
    def _dp(self, scores, lengths=None, force_grad=False):
        semiring = self.semiring
        ssize = semiring.size()
        batch, N, N, NT = scores.shape

        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        beta = self._make_chart(2, (batch, N, N, NT), rules, force_grad)
        span = self._make_chart(N, (batch, N, NT), rules, force_grad)
        rule_use = [
            self._make_chart(1, (batch, N - w - 1, NT, S, S), rules, force_grad)[0]
            for w in range(N - 1)
        ]
        ns = torch.arange(N)
        beta[A][:, :, 0, :] = scores[:, ns, ns]
        beta[B][:, :, N - 1,:] = scores[:, ns, ns]

        for w in range(1, N):
            Y = beta[A][:, :, : N - w, :w].view(ssize, batch, N - w, w, 1, NT, 1)
            Z = beta[B][:, :, w:, N - w :].view(ssize, batch, N - w, w, 1, 1, NT)

            rule_use[w - 1][:] = semiring.times(
                semiring.sum(semiring.times(Y, Z), dim=3), scores
            )
            span[w] = rule_use[w - 1].view(ssize, batch, N - w, NT)
            beta[A][:, :, : N - w, w] = span[w]
            beta[B][:, :, w:N, N - w - 1] = beta[A][:, :, : N - w, w]

        log_Z =  semiring.sum(torch.stack(
            [beta[A][:, b, 0, l - 1] for b, l in enumerate(lengths)], dim=1
        ), dim=1)
        return semiring.unconvert(log_Z), (rule_use), beta
