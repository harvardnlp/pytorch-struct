from .cky import cky_inside, cky, cky_check
from .deptree import deptree_inside, deptree, deptree_check, deptree_nonproj
from .linearchain import linearchain, linearchain_forward, linearchain_check, hmm
from .semimarkov import semimarkov, semimarkov_forward, semimarkov_check
from .semirings import LogSemiring, MaxSemiring, StdSemiring, SampledSemiring
import torch
from hypothesis import given, settings
from hypothesis.strategies import integers

smint = integers(min_value=2, max_value=4)
tint = integers(min_value=1, max_value=2)


@given(smint, smint, smint)
def test_simple(batch, N, C):
    vals = torch.ones(batch, N, C, C)
    semiring = StdSemiring

    alpha, _ = linearchain_forward(vals, semiring)
    assert (alpha == pow(C, N + 1)).all()

    linearchain(vals, SampledSemiring)


@given(smint, smint, tint)
@settings(max_examples=25)
def test_linearchain(batch, N, C):
    for semiring in [LogSemiring, MaxSemiring]:
        vals = torch.rand(batch, N, C, C)
        semiring = LogSemiring

        alpha, _ = linearchain_forward(vals, semiring)
        count = linearchain_check(vals, semiring)
        assert torch.isclose(count[0], alpha[0])

    vals = torch.rand(batch, N, C, C)
    semiring = MaxSemiring
    score, _ = linearchain_forward(vals, semiring)
    marginals = linearchain(vals, semiring)
    print(marginals.shape, vals.shape)
    assert torch.isclose(score.sum(), marginals.mul(vals).sum()).all()


def test_hmm():
    C, V, batch, N = 5, 20, 2, 5
    transition = torch.rand(C, C)
    emission = torch.rand(V, C)
    init = torch.rand(C)
    observations = torch.randint(0, V, (batch, N))
    out = hmm(transition, emission, init, observations)
    linearchain(out)


@given(smint, smint, smint, smint)
@settings(max_examples=50)
def test_semimarkov(N, K, V, C):
    batch = 2
    for semiring in [LogSemiring, MaxSemiring]:
        vals = torch.rand(batch, N, K, C, C)
        vals[:, :, 0, :, :] = semiring.zero()
        v, _ = semimarkov_forward(vals, semiring)
        count = semimarkov_check(vals, semiring)
        assert torch.isclose(v, count).all()
    vals = torch.rand(batch, N, K, C, C)
    semiring = MaxSemiring
    score, _ = semimarkov_forward(vals, semiring)
    marginals = semimarkov(vals, semiring)
    assert torch.isclose(score.sum(), marginals.mul(vals).sum()).all()


@given(smint)
@settings(max_examples=25)
def test_dep(N):
    batch = 2
    for semiring in [LogSemiring, MaxSemiring]:
        scores = torch.rand(batch, N, N)
        top, arcs = deptree_inside(scores, semiring)
        out = deptree_check(scores, semiring)
        assert torch.isclose(top[0], out)

    semiring = MaxSemiring
    score, _ = deptree_inside(scores, semiring)
    marginals = deptree(scores, semiring)
    assert torch.isclose(score.sum(), marginals.mul(scores).sum()).all()


def test_dep_np():
    N = 5
    batch = 2
    scores = torch.rand(batch, N, N)
    top, arcs = deptree_nonproj(scores)


@given(smint, tint, tint)
@settings(max_examples=50)
def test_cky(N, NT, T):
    batch = 2
    for semiring in [LogSemiring, MaxSemiring]:
        terms = torch.rand(batch, N, T)
        rules = torch.rand(batch, NT, (NT + T), (NT + T))
        roots = torch.rand(batch, NT)

        v2 = cky_check(terms, rules, roots, semiring)
        v, _ = cky_inside(terms, rules, roots, semiring)
        assert torch.isclose(v[0], v2)
    semiring = MaxSemiring
    score, _ = cky_inside(terms, rules, roots, semiring)
    (m_term, m_rule, m_root) = cky(terms, rules, roots, semiring)
    assert torch.isclose(
        score.sum(),
        (
            m_term.mul(terms).sum()
            + m_rule.sum(dim=1).sum(dim=1).mul(rules).sum()
            + m_root.mul(roots).sum()
        ).sum(),
    ).all()
