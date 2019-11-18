import torch
from .helpers import _Struct

A, B = 0, 1



class Get(torch.autograd.Function):
    @staticmethod
    def forward(ctx, chart, grad_chart, indices):
        out = chart[indices]
        ctx.save_for_backward(grad_chart)
        ctx.indices = indices
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_chart, = ctx.saved_tensors
        grad_chart[ctx.indices] = grad_output
        return grad_chart, None, None

# class Set(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, chart, grad_chart, indices, vals):
#         chart[indices] = vals
#         ctx.save_for_backward(grad_chart)
#         ctx.indices = indices
#         return None

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_chart, = ctx.saved_tensors
#         # grad_chart[ctx.indices] += grad_output
#         # size, = ctx.saved_tensors
#         # grad_chart = torch.zeros(*size.shape)
#         grad_chart[ctx.indices] = grad_output
#         return grad_chart, None, None


class CKY_CRF(_Struct):

    def _check_potentials(self, edge, lengths=None):
        batch, N, _, NT = edge.shape
        edge.requires_grad_(True)
        edge = self.semiring.convert(edge)
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        return edge, batch, N, NT, lengths

    def _dp(self, scores, lengths=None, force_grad=False):
        semiring = self.semiring
        scores, batch, N, NT, lengths = self._check_potentials(scores, lengths)
        beta = self._make_chart(2, (batch, N, N), scores, force_grad)
        grad_beta = self._make_chart(2, (batch, N, N), scores, force_grad)
        grad_beta[A].fill_(0.0)
        grad_beta[B].fill_(0.0)
        L_DIM, R_DIM = 2, 3

        # Initialize
        reduced_scores = semiring.sum(scores)
        term = reduced_scores.diagonal(0, L_DIM, R_DIM)
        ns = torch.arange(N)
        beta[A][:, :, ns, 0] = term
        beta[B][:, :, ns, N-1] = term

        ai = slice(None)
        # Run
        for w in range(1, N):
            # Y = beta[A][:, :, : N - w, :]
            # Z = beta[B][:, :, w:, :]
            # z = zero[:, :, : w]
            # score = reduced_scores.diagonal(w, L_DIM, R_DIM)
            # new = semiring.times(semiring.dot(Y, Z), score).unsqueeze(-1)
            # print(new.shape, z.shape, beta[A].shape)
            # beta[A] = c(beta[A], p(new, z))
            # beta[B] = c(p(z,  new), beta[B])

            Y = Get.apply(beta[A], grad_beta[A],
                          (ai, ai, slice(None, N - w), slice(None, w)))
            Z = Get.apply(beta[B], grad_beta[B],
                          (ai, ai, slice(w, None), slice(N-w, None)))
            score = reduced_scores.diagonal(w, L_DIM, R_DIM)
            new = semiring.times(semiring.dot(Y, Z), score)
            # Set.apply(beta[A], grad_beta[A], (ai, ai, slice(None, N - w), w), new)

            beta[A][(ai, ai, slice(None, N - w), w)] = new
            # beta[A][:, :, : N - w, w] =
            # beta[B][:, :, w:N, N - w - 1] =  beta[A][:, :, :N-w, w]

            beta[B][:, :, w:N, N - w - 1:N-w] =  beta[A][(ai, ai, slice(None, N-w), slice(w, w+1))]
                  # Get.apply(beta[A], grad_beta[A],
                  #          (ai, ai, slice(None, N-w), slice(w, w+1)))



        final = beta[A][:, :, 0]
        log_Z = final[:, torch.arange(batch), lengths - 1]
        return log_Z, [scores], beta

    # For testing

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
