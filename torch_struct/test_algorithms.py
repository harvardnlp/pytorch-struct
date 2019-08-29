from .cky import cky_inside, cky, cky_check
from .deptree import deptree_inside, deptree, deptree_check, deptree_nonproj
from .linearchain import linearchain, linearchain_forward, linearchain_check, hmm
from .semimarkov import semimarkov, semimarkov_forward, semimarkov_check
from .semirings import LogSemiring, StdSemiring, MaxSemiring
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
    for semiring in [LogSemiring, MaxSemiring]:
        vals = torch.rand(batch, N, C, C)
        semiring = LogSemiring

        alpha, _ = linearchain_forward(vals, semiring)
        count = linearchain_check(vals, semiring)
        print(count.shape, alpha.shape)
        assert torch.isclose(count[0], alpha[0])
        marginals = linearchain(vals, semiring)


def test_hmm():
    C, V, batch, N = 5, 20, 2, 5
    transition = torch.rand(C, C)
    emission = torch.rand(V, C)
    init = torch.rand(C)
    observations = torch.randint(0, V, (batch, N))
    out = hmm(transition, emission, init, observations)
    linearchain(out)


smint = integers(min_value=2, max_value=5)


def test_semimarkov():
    batch = 2
    N, K, C = 4, 3, 5
    for semiring in [LogSemiring, MaxSemiring]:
        vals = torch.rand(batch, N, K, C, C)
        vals[:, :, 0, :, :] = semiring.zero()
        v, _ = semimarkov_forward(vals, semiring)
        count = semimarkov_check(vals, semiring)
        assert torch.isclose(v, count).all()


def test_dep():
    N, batch = 5, 2
    for semiring in [LogSemiring, MaxSemiring]:
        scores = torch.rand(batch, N, N)
        top, arcs = deptree_inside(scores, semiring)
        out = deptree_check(scores, semiring)
        assert torch.isclose(top[0], out)


def test_dep_np():
    N = 5
    batch = 2
    scores = torch.rand(batch, N, N)
    top, arcs = deptree_nonproj(scores)


def test_cky():
    N = 3
    batch = 1
    NT = 4
    T = 5

    for semiring in [LogSemiring, MaxSemiring]:
        terms = torch.rand(batch, N, T)
        rules = torch.rand(batch, NT, (NT + T), (NT + T))
        roots = torch.rand(batch, NT)

        v2 = cky_check(terms, rules, roots, semiring)
        v, _ = cky_inside(terms, rules, roots, semiring)
        assert torch.isclose(v[0], v2)
