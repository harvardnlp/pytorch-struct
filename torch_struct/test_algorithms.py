from .cky import CKY
from .cky_crf import CKY_CRF
from .deptree import DepTree, deptree_nonproj, deptree_part
from .linearchain import LinearChain
from .semimarkov import SemiMarkov
from .alignment import Alignment
from .semirings import (
    LogSemiring,
    CheckpointSemiring,
    CheckpointShardSemiring,
    KMaxSemiring,
    SparseMaxSemiring,
    MaxSemiring,
    StdSemiring,
    SampledSemiring,
    EntropySemiring,
    MultiSampledSemiring,
)
import torch
from hypothesis import given, settings
from hypothesis.strategies import integers, data, sampled_from

smint = integers(min_value=2, max_value=4)
tint = integers(min_value=1, max_value=2)
lint = integers(min_value=2, max_value=10)


@given(smint, smint, smint)
@settings(max_examples=50, deadline=None)
def test_simple_a(batch, N, C):
    vals = torch.ones(batch, N, C, C)
    semiring = StdSemiring
    alpha = LinearChain(semiring).sum(vals)
    c = pow(C, N + 1)
    print(c)
    assert (alpha == c).all()
    LinearChain(SampledSemiring).marginals(vals)

    LinearChain(MultiSampledSemiring).marginals(vals)


@given(smint, smint, smint, smint)
@settings(max_examples=50, deadline=None)
def test_simple_b(batch, N, K, C):
    print(N)
    N = 14
    vals = torch.ones(batch, N, 5, C, C)
    SemiMarkov(SampledSemiring).marginals(vals)
    SemiMarkov(MultiSampledSemiring).marginals(vals)


# @given(data())
# @settings(max_examples=50, deadline=None)
# def test_networkx(data):
#     batch = 5
#     N = 10
#     NT = 5
#     T = 5

#     torch.manual_seed(0)

#     terms = torch.rand(batch, N, T)
#     rules = torch.rand(batch, NT, (NT + T), (NT + T))
#     roots = torch.rand(batch, NT)
#     vals = (terms, rules, roots)
#     model = CKY
#     lengths = torch.tensor(
#         [data.draw(integers(min_value=3, max_value=N)) for b in range(batch - 1)] + [N]
#     )
#     struct = model(SampledSemiring)
#     marginals = struct.marginals(vals, lengths=lengths)
#     spans = CKY.from_parts(marginals)[0]
#     CKY.to_networkx(spans)

#     struct = model(MultiSampledSemiring)
#     marginals = struct.marginals(vals, lengths=lengths)
#     m2 = tuple((MultiSampledSemiring.to_discrete(m, 5) for m in marginals))
#     spans = CKY.from_parts(m2)[0]
#     CKY.to_networkx(spans)


@given(data())
def test_entropy(data):
    model = data.draw(sampled_from([LinearChain, SemiMarkov]))
    semiring = EntropySemiring
    struct = model(semiring)
    vals, (batch, N) = model._rand()
    alpha = struct.sum(vals)

    log_z = model(LogSemiring).sum(vals)
    log_probs = model(LogSemiring).enumerate(vals)[1]
    log_probs = torch.stack(log_probs, dim=1) - log_z
    print(log_probs.shape, log_z.shape, log_probs.exp().sum(1))
    entropy = -log_probs.mul(log_probs.exp()).sum(1).squeeze(0)
    assert entropy.shape == alpha.shape
    assert torch.isclose(entropy, alpha).all()


@given(data())
def test_kmax(data):
    model = data.draw(sampled_from([LinearChain, SemiMarkov, DepTree]))
    K = 2
    semiring = KMaxSemiring(K)
    struct = model(semiring)
    vals, (batch, N) = model._rand()
    max1 = model(MaxSemiring).sum(vals)
    alpha = struct.sum(vals, _raw=True)
    assert (alpha[0] == max1).all()
    assert (alpha[1] <= max1).all()

    topk = struct.marginals(vals, _raw=True)
    argmax = model(MaxSemiring).marginals(vals)
    assert (topk[0] == argmax).all()
    print(topk[0].nonzero(), topk[1].nonzero())
    assert (topk[1] != topk[0]).any()

    if model != DepTree:
        log_probs = model(MaxSemiring).enumerate(vals)[1]
        tops = torch.topk(torch.cat(log_probs, dim=0), 5, 0)[0]
        assert torch.isclose(struct.score(topk[1], vals), alpha[1]).all()
        for k in range(K):
            assert (torch.isclose(alpha[k], tops[k])).all()


@given(data())
@settings(max_examples=50, deadline=None)
def test_cky(data):
    model = data.draw(sampled_from([CKY]))
    semiring = data.draw(sampled_from([LogSemiring, MaxSemiring]))
    struct = model(semiring)
    vals, (batch, N) = model._rand()
    alpha = struct.sum(vals)
    count = struct.enumerate(vals)[0]

    assert alpha.shape[0] == batch
    assert count.shape[0] == batch
    assert alpha.shape == count.shape
    assert torch.isclose(count[0], alpha[0])


@given(data())
@settings(max_examples=50, deadline=None)
def test_generic_a(data):
    model = data.draw(
        sampled_from(
            [SemiMarkov]
        )  # , Alignment , LinearChain, SemiMarkov, CKY, CKY_CRF, DepTree])
    )

    semiring = data.draw(sampled_from([LogSemiring, MaxSemiring]))
    struct = model(semiring)
    vals, (batch, N) = model._rand()
    alpha = struct.sum(vals)
    count = struct.enumerate(vals)[0]
    # assert(False)
    assert alpha.shape[0] == batch
    assert count.shape[0] == batch
    assert alpha.shape == count.shape
    assert torch.isclose(count[0], alpha[0])

    vals, _ = model._rand()
    struct = model(MaxSemiring)
    score = struct.sum(vals)
    marginals = struct.marginals(vals)
    # print(marginals)
    # # assert(False)
    assert torch.isclose(score, struct.score(vals, marginals)).all()


@given(data())
@settings(max_examples=50, deadline=None)
def test_labeled_proj_deptree(data):
    semiring = data.draw(sampled_from([LogSemiring, MaxSemiring]))
    struct = DepTree(semiring)
    arc_scores = torch.rand(3, 5, 5, 7)
    count = struct.enumerate(semiring.sum(arc_scores))[0]
    alpha = struct.sum(arc_scores)

    assert torch.isclose(count, alpha).all()

    struct = DepTree(MaxSemiring)
    max_score = struct.sum(arc_scores)
    argmax = struct.marginals(arc_scores)
    assert torch.isclose(max_score, struct.score(arc_scores, argmax)).all()


@given(data())
@settings(max_examples=50, deadline=None)
def test_non_proj(data):
    model = data.draw(sampled_from([DepTree]))
    semiring = data.draw(sampled_from([LogSemiring]))
    struct = model(semiring)
    vals, (batch, N) = model._rand()
    alpha = deptree_part(vals)
    count = struct.enumerate(vals, non_proj=True, multi_root=False)[0]

    assert alpha.shape[0] == batch
    assert count.shape[0] == batch
    assert alpha.shape == count.shape
    assert torch.isclose(count[0], alpha[0])

    marginals = deptree_nonproj(vals)
    print(marginals.sum(1))
    # assert(False)
    # vals, _ = model._rand()
    # struct = model(MaxSemiring)
    # score = struct.sum(vals)
    # marginals = struct.marginals(vals)
    # assert torch.isclose(score, struct.score(vals, marginals)).all()


@given(data(), integers(min_value=1, max_value=20))
def test_parts_from_marginals(data, seed):
    # todo: add CKY, DepTree too?
    model = data.draw(sampled_from([LinearChain, SemiMarkov]))
    struct = model()
    torch.manual_seed(seed)
    vals, (batch, N) = struct._rand()

    edge = model(MaxSemiring).marginals(vals).long()

    sequence, extra = model.from_parts(edge)
    edge_ = model.to_parts(sequence, extra)

    assert (torch.isclose(edge, edge_)).all(), edge - edge_

    sequence_, extra_ = model.from_parts(edge_)
    assert extra == extra_, (extra, extra_)

    assert (torch.isclose(sequence, sequence_)).all(), sequence - sequence_


@given(data(), integers(min_value=1, max_value=20))
def test_parts_from_sequence(data, seed):
    model = data.draw(sampled_from([LinearChain, SemiMarkov]))
    struct = model()
    torch.manual_seed(seed)
    vals, (batch, N) = struct._rand()
    C = vals.size(-1)
    if isinstance(struct, LinearChain):
        K = 2
        background = 0
        extra = C
    elif isinstance(struct, SemiMarkov):
        K = vals.size(-3)
        background = -1
        extra = C, K
    else:
        raise NotImplementedError()

    sequence = torch.full((batch, N), background).long()
    for b in range(batch):
        i = 0
        while i < N:
            symbol = torch.randint(0, C, (1,)).item()
            sequence[b, i] = symbol
            length = torch.randint(1, K, (1,)).item()
            i += length

    edge = model.to_parts(sequence, extra)
    sequence_, extra_ = model.from_parts(edge)
    assert extra == extra_, (extra, extra_)
    assert (torch.isclose(sequence, sequence_)).all(), sequence - sequence_
    edge_ = model.to_parts(sequence_, extra_)
    assert (torch.isclose(edge, edge_)).all(), edge - edge_


@given(data(), integers(min_value=1, max_value=10))
@settings(max_examples=50, deadline=None)
def test_generic_lengths(data, seed):
    model = data.draw(
        sampled_from([CKY, Alignment, LinearChain, SemiMarkov, CKY_CRF, DepTree])
    )
    struct = model()
    torch.manual_seed(seed)
    vals, (batch, N) = struct._rand()
    lengths = torch.tensor(
        [data.draw(integers(min_value=2, max_value=N)) for b in range(batch - 1)] + [N]
    )

    m = model(MaxSemiring).marginals(vals, lengths=lengths)
    maxes = struct.score(vals, m)
    part = model().sum(vals, lengths=lengths)
    print(maxes, part)
    assert (maxes <= part).all()
    m_part = model(MaxSemiring).sum(vals, lengths=lengths)
    assert (torch.isclose(maxes, m_part)).all(), maxes - m_part

    # m2 = deptree(vals, lengths=lengths)
    # assert (m2 < part).all()

    if model == CKY:
        return

    seqs, extra = struct.from_parts(m)
    # assert (seqs.shape == (batch, N))
    # assert seqs.max().item() <= N
    full = struct.to_parts(seqs, extra, lengths=lengths)

    if isinstance(full, tuple):
        for i in range(len(full)):
            if i == 1:
                p = m[i].sum(1).sum(1)
            else:
                p = m[i]
            assert (full[i] == p.type_as(full[i])).all(), "%s %s %s" % (
                i,
                full[i].nonzero(),
                p.nonzero(),
            )
    else:
        assert (full == m.type_as(full)).all(), "%s %s %s" % (
            full.shape,
            m.shape,
            (full - m.type_as(full)).nonzero(),
        )


@settings(max_examples=50, deadline=None)
@given(data(), integers(min_value=1, max_value=10))
def test_params(data, seed):
    model = data.draw(
        sampled_from([Alignment, DepTree, SemiMarkov, DepTree, CKY, CKY_CRF])
    )
    struct = model()
    torch.manual_seed(seed)
    vals, (batch, N) = struct._rand()
    if isinstance(vals, tuple):
        vals = tuple((v.requires_grad_(True) for v in vals))
    else:
        vals.requires_grad_(True)
    # torch.autograd.set_detect_anomaly(True)
    semiring = LogSemiring
    alpha = model(semiring).sum(vals)
    alpha.sum().backward()

    if not isinstance(vals, tuple):
        b = vals.grad.detach()
        vals.grad.zero_()
        alpha = model(semiring).sum(vals, _autograd=False)
        alpha.sum().backward()
        c = vals.grad.detach()
        assert torch.isclose(b, c).all()


@given(data())
@settings(max_examples=50, deadline=None)
def test_alignment(data):

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
    struct = model(semiring, sparse_rounds=10)
    vals, (batch, N) = model._rand()
    alpha = struct.sum(vals)
    count = struct.enumerate(vals)[0]
    assert torch.isclose(count, alpha).all()

    model = data.draw(sampled_from([Alignment]))
    semiring = data.draw(sampled_from([LogSemiring]))
    struct = model(semiring, sparse_rounds=10)
    vals, (batch, N) = model._rand()
    alpha = struct.sum(vals)
    count = struct.enumerate(vals)[0]
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
    vals, (batch, N) = model._rand()
    vals[..., 0] = -2 * vals[..., 0].abs()
    vals[..., 1] = vals[..., 1].abs()
    vals[..., 2] = -2 * vals[..., 2].abs()
    alpha = struct.sum(vals)
    count = struct.enumerate(vals)[0]
    mx = struct.marginals(vals)
    print(alpha, count)
    print(mx[0].nonzero())
    # assert torch.isclose(count, alpha).all()
    struct = model(semiring, max_gap=1)
    alpha = struct.sum(vals)


def test_hmm():
    C, V, batch, N = 5, 20, 2, 5
    transition = torch.rand(C, C)
    emission = torch.rand(V, C)
    init = torch.rand(C)
    observations = torch.randint(0, V, (batch, N))
    out = LinearChain.hmm(transition, emission, init, observations)
    LinearChain().sum(out)


@given(data())
def test_sparse_max(data):
    model = data.draw(sampled_from([LinearChain]))
    semiring = SparseMaxSemiring
    vals, (batch, N) = model._rand()
    vals.requires_grad_(True)
    model(semiring).sum(vals)
    sparsemax = model(semiring).marginals(vals)
    print(vals.requires_grad)
    sparsemax.sum().backward()


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
