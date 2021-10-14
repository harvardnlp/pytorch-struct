from torch_struct import (
    CKY,
    CKY_CRF,
    DepTree,
    LinearChain,
    SemiMarkov,
    Alignment,
    deptree_nonproj,
    deptree_part,
)
from torch_struct import (
    LogSemiring,
    CheckpointSemiring,
    CheckpointShardSemiring,
    KMaxSemiring,
    SparseMaxSemiring,
    MaxSemiring,
    StdSemiring,
    EntropySemiring,
)
from .extensions import (
    LinearChainTest,
    SemiMarkovTest,
    DepTreeTest,
    CKYTest,
    CKY_CRFTest,
    test_lookup,
)
import torch
from hypothesis import given
from hypothesis.strategies import integers, data, sampled_from
import pytest

from hypothesis import settings

settings.register_profile("ci", max_examples=50, deadline=None)

settings.load_profile("ci")


smint = integers(min_value=2, max_value=4)
tint = integers(min_value=1, max_value=2)
lint = integers(min_value=2, max_value=10)


algorithms = {
    "LinearChain": (LinearChain, LinearChainTest),
    "SemiMarkov": (SemiMarkov, SemiMarkovTest),
    "Dep": (DepTree, DepTreeTest),
    "CKY_CRF": (CKY_CRF, CKY_CRFTest),
    "CKY": (CKY, CKYTest),
}


class Gen:
    "Helper class for tests"

    def __init__(self, model_test, data, semiring):
        model_test = algorithms[model_test]
        self.data = data
        self.model = model_test[0]
        self.struct = self.model(semiring)
        self.test = model_test[1]
        self.vals, (self.batch, self.N) = data.draw(self.test.logpotentials())
        # jitter
        if not isinstance(self.vals, tuple):
            self.vals = self.vals + 1e-6 * torch.rand(*self.vals.shape)
        self.semiring = semiring

    def enum(self, semiring=None):
        return self.test.enumerate(
            semiring if semiring is not None else self.semiring, self.vals
        )


# Model specific tests.


@given(smint, smint, smint)
@settings(max_examples=50, deadline=None)
def test_linear_chain_counting(batch, N, C):
    vals = torch.ones(batch, N, C, C)
    semiring = StdSemiring
    alpha = LinearChain(semiring).sum(vals)
    c = pow(C, N + 1)
    assert (alpha == c).all()


# Semiring tests


@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain", "SemiMarkov", "Dep"])
@pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring])
def test_log_shapes(model_test, semiring, data):
    gen = Gen(model_test, data, semiring)
    alpha = gen.struct.sum(gen.vals)
    count = gen.enum()[0]

    assert alpha.shape[0] == gen.batch
    assert count.shape[0] == gen.batch
    assert alpha.shape == count.shape
    assert torch.isclose(count[0], alpha[0])


@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain", "SemiMarkov"])
def test_entropy(model_test, data):
    "Test entropy by manual enumeration"
    gen = Gen(model_test, data, EntropySemiring)
    alpha = gen.struct.sum(gen.vals)
    log_z = gen.model(LogSemiring).sum(gen.vals)

    log_probs = gen.enum(LogSemiring)[1]
    log_probs = torch.stack(log_probs, dim=1) - log_z
    entropy = -log_probs.mul(log_probs.exp()).sum(1).squeeze(0)
    assert entropy.shape == alpha.shape
    assert torch.isclose(entropy, alpha).all()


@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain"])
def test_sparse_max(model_test, data):
    gen = Gen(model_test, data, SparseMaxSemiring)
    gen.vals.requires_grad_(True)
    gen.struct.sum(gen.vals)
    sparsemax = gen.struct.marginals(gen.vals)
    sparsemax.sum().backward()


@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain", "SemiMarkov", "Dep"])
def test_kmax(model_test, data):
    "Test out the k-max semiring"
    K = 2
    gen = Gen(model_test, data, KMaxSemiring(K))
    max1 = gen.model(MaxSemiring).sum(gen.vals)
    alpha = gen.struct.sum(gen.vals, _raw=True)

    # 2max is less than max.
    assert (alpha[0] == max1).all()
    assert (alpha[1] <= max1).all()

    topk = gen.struct.marginals(gen.vals, _raw=True)
    argmax = gen.model(MaxSemiring).marginals(gen.vals)

    # Argmax is different than 2-argmax
    assert (topk[0] == argmax).all()
    assert (topk[1] != topk[0]).any()

    if model_test != "Dep":
        log_probs = gen.enum(MaxSemiring)[1]
        tops = torch.topk(torch.cat(log_probs, dim=0), 5, 0)[0]
        assert torch.isclose(gen.struct.score(topk[1], gen.vals), alpha[1]).all()
        for k in range(K):
            assert (torch.isclose(alpha[k], tops[k])).all()


@given(data())
@pytest.mark.parametrize("model_test", ["CKY"])
@pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring])
def test_cky(model_test, semiring, data):
    gen = Gen(model_test, data, semiring)
    alpha = gen.struct.sum(gen.vals)
    count = gen.enum()[0]

    assert alpha.shape[0] == gen.batch
    assert count.shape[0] == gen.batch
    assert alpha.shape == count.shape
    assert torch.isclose(count[0], alpha[0])


@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain", "SemiMarkov", "CKY_CRF", "Dep"])
def test_max(model_test, data):
    "Test that argmax score is the same as max"
    gen = Gen(model_test, data, MaxSemiring)
    score = gen.struct.sum(gen.vals)
    marginals = gen.struct.marginals(gen.vals)
    assert torch.isclose(score, gen.struct.score(gen.vals, marginals)).all()


@given(data())
@pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring])
@pytest.mark.parametrize("model_test", ["Dep"])
def test_labeled_proj_deptree(model_test, semiring, data):
    gen = Gen(model_test, data, semiring)

    arc_scores = torch.rand(3, 5, 5, 7)
    gen.vals = semiring.sum(arc_scores)
    count = gen.enum()[0]
    alpha = gen.struct.sum(arc_scores)

    assert torch.isclose(count, alpha).all()

    struct = gen.model(MaxSemiring)
    max_score = struct.sum(arc_scores)
    argmax = struct.marginals(arc_scores)
    assert torch.isclose(max_score, struct.score(arc_scores, argmax)).all()


# todo: add CKY, DepTree too?
@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain", "SemiMarkov", "Dep", "CKY_CRF"])
def test_parts_from_marginals(model_test, data):
    gen = Gen(model_test, data, MaxSemiring)

    edge = gen.struct.marginals(gen.vals).long()
    sequence, extra = gen.model.from_parts(edge)
    edge_ = gen.model.to_parts(sequence, extra)

    assert (torch.isclose(edge, edge_)).all(), edge - edge_

    sequence_, extra_ = gen.model.from_parts(edge_)
    assert extra == extra_, (extra, extra_)
    assert (torch.isclose(sequence, sequence_)).all(), sequence - sequence_


@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain", "SemiMarkov"])
def test_parts_from_sequence(model_test, data):
    gen = Gen(model_test, data, LogSemiring)
    C = gen.vals.size(-1)
    if isinstance(gen.struct, LinearChain):
        K = 2
        background = 0
        extra = C
    elif isinstance(gen.struct, SemiMarkov):
        K = gen.vals.size(-3)
        background = -1
        extra = C, K
    else:
        raise NotImplementedError()

    sequence = torch.full((gen.batch, gen.N), background, dtype=int)
    for b in range(gen.batch):
        i = 0
        while i < gen.N:
            symbol = torch.randint(0, C, (1,)).item()
            sequence[b, i] = symbol
            length = torch.randint(1, K, (1,)).item()
            i += length

    edge = gen.model.to_parts(sequence, extra)
    sequence_, extra_ = gen.model.from_parts(edge)
    assert extra == extra_, (extra, extra_)
    assert (torch.isclose(sequence, sequence_)).all(), sequence - sequence_
    edge_ = gen.model.to_parts(sequence_, extra_)
    assert (torch.isclose(edge, edge_)).all(), edge - edge_


@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain", "SemiMarkov", "CKY_CRF", "Dep"])
def test_generic_lengths(model_test, data):
    gen = Gen(model_test, data, LogSemiring)
    model, struct, vals, N, batch = gen.model, gen.struct, gen.vals, gen.N, gen.batch
    lengths = torch.tensor(
        [data.draw(integers(min_value=2, max_value=N)) for b in range(batch - 1)] + [N]
    )

    m = model(MaxSemiring).marginals(vals, lengths=lengths)
    maxes = struct.score(vals, m)
    part = model().sum(vals, lengths=lengths)

    # Check that max is correct
    assert (maxes <= part + 1e-3).all()
    m_part = model(MaxSemiring).sum(vals, lengths=lengths)
    assert (torch.isclose(maxes, m_part)).all(), maxes - m_part

    if model == CKY:
        return

    seqs, extra = struct.from_parts(m)
    full = struct.to_parts(seqs, extra, lengths=lengths)

    assert (full == m.type_as(full)).all(), "%s %s %s" % (
        full.shape,
        m.shape,
        (full - m.type_as(full)).nonzero(),
    )


@given(data())
@pytest.mark.parametrize(
    "model_test", ["LinearChain", "SemiMarkov", "Dep", "CKY", "CKY_CRF"]
)
def test_params(model_test, data):
    gen = Gen(model_test, data, LogSemiring)
    _, struct, vals, _, _ = gen.model, gen.struct, gen.vals, gen.N, gen.batch

    if isinstance(vals, tuple):
        vals = tuple((v.requires_grad_(True) for v in vals))
    else:
        vals.requires_grad_(True)
    alpha = struct.sum(vals)
    alpha.sum().backward()


@given(data())
@pytest.mark.parametrize("model_test", ["LinearChain", "SemiMarkov", "Dep"])
def test_gumbel(model_test, data):
    gen = Gen(model_test, data, LogSemiring)
    gen.vals.requires_grad_(True)
    alpha = gen.struct.marginals(gen.vals)
    print(alpha[0])
    print(torch.autograd.grad(alpha, gen.vals, alpha.detach())[0][0])


def test_hmm():
    C, V, batch, N = 5, 20, 2, 5
    transition = torch.rand(C, C)
    emission = torch.rand(V, C)
    init = torch.rand(C)
    observations = torch.randint(0, V, (batch, N))
    out = LinearChain.hmm(transition, emission, init, observations)
    LinearChain().sum(out)


def test_sparse_max2():
    print(LinearChain(SparseMaxSemiring).sum(torch.rand(1, 8, 3, 3)))
    print(LinearChain(SparseMaxSemiring).marginals(torch.rand(1, 8, 3, 3)))
    # assert(False)


def test_lc_custom():
    model = LinearChain
    vals, _ = model._rand()

    struct = LinearChain(LogSemiring)
    marginals = struct.marginals(vals)
    s = struct.sum(vals)

    struct = LinearChain(CheckpointSemiring(LogSemiring, 1))
    marginals2 = struct.marginals(vals)
    s2 = struct.sum(vals)
    assert torch.isclose(s, s2).all()
    assert torch.isclose(marginals, marginals2).all()

    struct = LinearChain(CheckpointShardSemiring(LogSemiring, 1))
    marginals2 = struct.marginals(vals)
    s2 = struct.sum(vals)
    assert torch.isclose(s, s2).all()
    assert torch.isclose(marginals, marginals2).all()

    # struct = LinearChain(LogMemSemiring)
    # marginals2 = struct.marginals(vals)
    # s2 = struct.sum(vals)
    # assert torch.isclose(s, s2).all()
    # assert torch.isclose(marginals, marginals).all()

    # struct = LinearChain(LogMemSemiring)
    # marginals = struct.marginals(vals)
    # s = struct.sum(vals)

    # struct = LinearChain(LogSemiringKO)
    # marginals2 = struct.marginals(vals)
    # s2 = struct.sum(vals)
    # assert torch.isclose(s, s2).all()
    # assert torch.isclose(marginals, marginals).all()
    # print(marginals)
    # print(marginals2)

    # struct = LinearChain(LogSemiring)
    # marginals = struct.marginals(vals)
    # s = struct.sum(vals)

    # struct = LinearChain(LogSemiringKO)
    # marginals2 = struct.marginals(vals)
    # s2 = struct.sum(vals)
    # assert torch.isclose(s, s2).all()
    # print(marginals)
    # print(marginals2)

    # struct = LinearChain(MaxSemiring)
    # marginals = struct.marginals(vals)
    # s = struct.sum(vals)

    # struct = LinearChain(MaxSemiringKO)
    # marginals2 = struct.marginals(vals)
    # s2 = struct.sum(vals)
    # assert torch.isclose(s, s2).all()
    # assert torch.isclose(marginals, marginals2).all()


@given(data())
@pytest.mark.parametrize("model_test", ["Dep"])
@pytest.mark.parametrize("semiring", [LogSemiring])
def test_non_proj(model_test, semiring, data):
    gen = Gen(model_test, data, semiring)
    alpha = deptree_part(gen.vals, False)
    count = gen.test.enumerate(LogSemiring, gen.vals, non_proj=True, multi_root=False)[
        0
    ]

    assert alpha.shape[0] == gen.batch
    assert count.shape[0] == gen.batch
    assert alpha.shape == count.shape
    # assert torch.isclose(count[0], alpha[0], 1e-2)

    alpha = deptree_part(gen.vals, True)
    count = gen.test.enumerate(LogSemiring, gen.vals, non_proj=True, multi_root=True)[0]

    assert alpha.shape[0] == gen.batch
    assert count.shape[0] == gen.batch
    assert alpha.shape == count.shape
    # assert torch.isclose(count[0], alpha[0], 1e-2)

    marginals = deptree_nonproj(gen.vals, multi_root=False)
    print(marginals.sum(1))
    marginals = deptree_nonproj(gen.vals, multi_root=True)
    print(marginals.sum(1))


#     # assert(False)
#     # vals, _ = model._rand()
#     # struct = model(MaxSemiring)
#     # score = struct.sum(vals)
#     # marginals = struct.marginals(vals)
#     # assert torch.isclose(score, struct.score(vals, marginals)).all()


@given(data())
@settings(max_examples=50, deadline=None)
def ignore_alignment(data):

    # log_potentials = torch.ones(2, 2, 2, 3)
    # v = Alignment(StdSemiring).sum(log_potentials)
    # print("FINAL", v)
    # log_potentials = torch.ones(2, 3, 2, 3)
    # v = Alignment(StdSemiring).sum(log_potentials)
    # print("FINAL", v)

    # log_potentials = torch.ones(2, 6, 2, 3)
    # v = Alignment(StdSemiring).sum(log_potentials)
    # print("FINAL", v)

    # log_potentials = torch.ones(2, 7, 2, 3)
    # v = Alignment(StdSemiring).sum(log_potentials)
    # print("FINAL", v)

    # log_potentials = torch.ones(2, 8, 2, 3)
    # v = Alignment(StdSemiring).sum(log_potentials)
    # print("FINAL", v)
    # assert False

    # model = data.draw(sampled_from([Alignment]))
    # semiring = data.draw(sampled_from([StdSemiring]))
    # struct = model(semiring)
    # vals, (batch, N) = model._rand()
    # print(batch, N)
    # struct = model(semiring)
    # # , max_gap=max(3, abs(vals.shape[1] - vals.shape[2]) + 1))
    # vals.fill_(1)
    # alpha = struct.sum(vals)

    model = data.draw(sampled_from([Alignment]))
    semiring = data.draw(sampled_from([StdSemiring]))
    test = test_lookup[model](semiring)
    struct = model(semiring, sparse_rounds=10)
    vals, (batch, N) = test._rand()
    alpha = struct.sum(vals)
    count = test.enumerate(vals)[0]
    assert torch.isclose(count, alpha).all()

    model = data.draw(sampled_from([Alignment]))
    semiring = data.draw(sampled_from([LogSemiring]))
    struct = model(semiring, sparse_rounds=10)
    vals, (batch, N) = model._rand()
    alpha = struct.sum(vals)
    count = test_lookup[model](semiring).enumerate(vals)[0]
    assert torch.isclose(count, alpha).all()

    # model = data.draw(sampled_from([Alignment]))
    # semiring = data.draw(sampled_from([MaxSemiring]))
    # struct = model(semiring)
    # log_potentials = torch.ones(2, 2, 2, 3)
    # v = Alignment(StdSemiring).sum(log_potentials)

    log_potentials = torch.ones(2, 2, 8, 3)
    v = Alignment(MaxSemiring).sum(log_potentials)
    # print(v)
    # assert False
    m = Alignment(MaxSemiring).marginals(log_potentials)
    score = Alignment(MaxSemiring).score(log_potentials, m)
    assert torch.isclose(v, score).all()

    semiring = data.draw(sampled_from([MaxSemiring]))
    struct = model(semiring, local=True)
    test = test_lookup[model](semiring)
    vals, (batch, N) = test._rand()
    vals[..., 0] = -2 * vals[..., 0].abs()
    vals[..., 1] = vals[..., 1].abs()
    vals[..., 2] = -2 * vals[..., 2].abs()
    alpha = struct.sum(vals)
    count = test.enumerate(vals)[0]
    mx = struct.marginals(vals)
    print(alpha, count)
    print(mx[0].nonzero())
    # assert torch.isclose(count, alpha).all()
    struct = model(semiring, max_gap=1)
    alpha = struct.sum(vals)


@pytest.mark.parametrize("model_test", ["SemiMarkov"])
@pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring])
def test_hsmm(model_test, semiring):
    "Test HSMM helper function."
    C, K, batch, N = 5, 3, 2, 5
    init_z_1 = torch.rand(batch, C)
    transition_z_to_z = torch.rand(C, C)
    transition_z_to_l = torch.rand(C, K)
    emission_n_l_z = torch.rand(batch, N, K, C)

    # first way: enumerate using init/transitions/emission
    partition1 = algorithms[model_test][1].enumerate_hsmm(semiring, init_z_1, transition_z_to_z,
                                                          transition_z_to_l, emission_n_l_z)[0]
    # second way: enumerate using edge scores computed from init/transitions/emission
    edge = SemiMarkov.hsmm(init_z_1, transition_z_to_z, transition_z_to_l, emission_n_l_z)
    partition2 = algorithms[model_test][1].enumerate(semiring, edge)[0]
    # third way: dp using edge scores computed from init/transitions/emission
    partition3 = algorithms[model_test][0](semiring).logpartition(edge)[0]
    # fourth way: dp_standard using edge scores computed from init/transitions/emission
    partition4 = algorithms[model_test][0](semiring)._dp_standard(edge)[0]

    assert torch.isclose(partition1, partition2).all()
    assert torch.isclose(partition2, partition3).all()
    assert torch.isclose(partition3, partition4).all()


@given(data())
@pytest.mark.parametrize("model_test", ["SemiMarkov"])
@pytest.mark.parametrize("semiring", [LogSemiring, MaxSemiring])
def test_batching_lengths(model_test, semiring, data):
    "Test batching"
    gen = Gen(model_test, data, LogSemiring)
    model, vals, N, batch = gen.model, gen.vals, gen.N, gen.batch
    lengths = torch.tensor(
        [data.draw(integers(min_value=2, max_value=N)) for b in range(batch - 1)] + [N]
    )
    # first way: batched implementation
    partition = model(semiring).logpartition(vals, lengths=lengths)[0][0]
    # second way: unbatched implementation
    for b in range(batch):
        vals_b = vals[b:(b + 1), :(lengths[b] - 1)]
        lengths_b = lengths[b:(b + 1)]
        partition_b = model(semiring).logpartition(vals_b, lengths=lengths_b)[0][0]
        assert torch.isclose(partition[b], partition_b).all()
    # test _dp_standard
    partition_dp_standard = model(semiring)._dp_standard(vals, lengths=lengths)[0][0]
    assert torch.isclose(partition, partition_dp_standard).all()
