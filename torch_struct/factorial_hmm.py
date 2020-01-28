import torch
from .helpers import _Struct, Chart
import math


class FactorialHMM(_Struct):
    def _dp(self, scores, lengths=None, force_grad=False):
        transition, emission = scores
        semiring = self.semiring
        transition.requires_grad_(True)
        emission.requires_grad_(True)
        batch, L, K, K2 = transition.shape
        batch, N, K, K, K = emission.shape
        assert L == 3
        assert K == K2

        transition = semiring.convert(transition)
        emission = semiring.convert(emission)


        ssize = semiring.size()

        state_out = Chart((batch, N, L, K), transition, semiring)
        state_in = Chart((batch, N, L, K), transition, semiring)
        emit = Chart((batch, N, K, K, K), transition, semiring)

        emit[0, :] = emission[:, :, 0]

        def make_out(val, i):
            state_out[i, 0] = semiring.sum(semiring.sum(val, 4), 2)
            state_out[i, 1] = semiring.sum(semiring.sum(val, 4), 3)
            state_out[i, 2] = semiring.sum(semiring.sum(val, 3), 2)

        make_out(emit[0, :], 0)

        for i in range(1, N):
            # print(transition.shape, state_out[i-1, :].unsqueeze(-2).shape)
            state_in = semiring.dot(state_out[i-1, :].unsqueeze(-2), transition)
            # print(state_in[..., None, :, None].shape,  emission[:, :, i].shape)
            emit[i, :] = semiring.times(state_in[..., 0, :, None, None],
                                        state_in[..., 1, None, :, None],
                                        state_in[..., 2, None, None, :],
                                        emission[:, :, i])
            make_out(emit[i, :], i)

        log_Z = semiring.sum(emit[N-1, :])
        return log_Z, [scores], None

    @staticmethod
    def _rand():
        batch = torch.randint(2, 5, (1,))
        K = torch.randint(2, 5, (1,))
        N = torch.randint(2, 5, (1,))
        transition = torch.rand(batch, 3, K, K)
        emission = torch.rand(batch, N, K, K, K)
        return (transition, emission), (batch.item(), N.item())
