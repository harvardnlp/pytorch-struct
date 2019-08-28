from .cky import cky_inside, cky, cky_check
from .deptree import deptree_inside, deptree, deptree_check
from .linearchain import  linearchain, linearchain_inside, linearchain_check
from .semimarkov import  semimarkov, semimarkov_inside, semimarkov_check
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
# @given(smint, smint, smint)
def test_linearchain():
    batch, N, C = 2, 3, 4
    # vals = torch.ones(batch, n, c, c)
    # semiring = StdSemiring
    vals = torch.rand(batch, N, C, C)
    semiring = LogSemiring

    alpha = linearchain_inside(vals, semiring)
    count = linearchain_check(vals, semiring)
    print(count.shape, alpha.shape)
    assert(torch.isclose(count[0], alpha[0]))


smint = integers(min_value=2, max_value=5)
def test_semimarkov():
    batch = 2
    N, K, C = 4, 3, 5
    # vals = torch.ones(batch, n, K, C, C)
    # semiring = StdSemiring
    vals = torch.rand(batch, N, K, C, C)
    semiring = LogSemiring
    vals[:, :, 0, :, :] = semiring.zero()
    v = semimarkov_inside(vals, semiring)
    count = semimarkov_check(vals, semiring)
    assert(torch.isclose(v, count).all())



@given(smint, smint)
def test_dep(batch, N):
    N = 5
    batch = 2

    scores = torch.rand(batch, N, N)
    semiring = LogSemiring

    top, arcs = deptree_inside(scores, semiring)
    out = deptree_check(scores, semiring)
    assert(torch.isclose(top[0], out))


def test_cky():
    N = 3
    batch = 1
    NT = 4
    T = 5
    terms = torch.rand(batch, N, T)
    rules = torch.rand(batch, NT, (NT+T), (NT+T))
    roots = torch.rand(batch, NT)
    semiring = LogSemiring

    v2 = cky_check(terms, rules, roots, semiring)
    v, _ = cky_inside(terms, rules, roots, semiring)
    assert(torch.isclose(v[0], v2))
