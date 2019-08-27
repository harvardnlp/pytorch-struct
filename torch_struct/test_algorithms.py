from .algorithms import linearchain, linearchain_inside, linearchain_check, dependencytree_inside

from .semirings import LogSemiring, StdSemiring
import torch
import math
from hypothesis import given
from hypothesis.strategies import (
    sampled_from,
    lists,
    data,
    floats,
    integers,
    permutations,
)

smint = integers(min_value=2, max_value=5)
@given(smint, smint, smint)
def test_chain(batch, n, c):
    total_paths = math.pow(c, n + 1)
    alpha = linearchain_inside(torch.zeros(batch, n, c, c).exp().float(),
                        semiring=StdSemiring)
    count = linearchain_check(torch.zeros(batch, n, c, c).exp().float(),
                        semiring=StdSemiring)
    print(count.shape, alpha[n].shape)
    assert(torch.isclose(StdSemiring.sum(alpha[n]), count).all())


@given(smint, smint, smint)
def test_dep(batch, n, c):
    pass
