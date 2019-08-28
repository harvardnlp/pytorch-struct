import torch
from .semirings import LogSemiring
import opt_einsum as oe
import itertools
"""
Conventions:
 > N -> length
 > C -> states
"""


def dependencytree_nonproj_marginals(arcs, eps=1e-5):
    laplacian = input.exp() + self.eps
    output = input.clone()
    for b in range(input.size(0)):
        lap = laplacian[b].masked_fill(
            torch.eye(input.size(1), device=input.device) != 0, 0
        )
        lap = -lap + torch.diag(lap.sum(0))
        # store roots on diagonal
        lap[0] = input[b].diag().exp()
        inv_laplacian = lap.inverse()
        factor = inv_laplacian.diag().unsqueeze(1).expand_as(input[b]).transpose(0, 1)
        term1 = input[b].exp().mul(factor).clone()
        term2 = input[b].exp().mul(inv_laplacian.transpose(0, 1)).clone()
        term1[:, 0] = 0
        term2[0] = 0
        output[b] = term1 - term2
        roots_output = input[b].diag().exp().mul(inv_laplacian.transpose(0, 1)[0])
        output[b] = output[b] + torch.diag(roots_output)
    return output
