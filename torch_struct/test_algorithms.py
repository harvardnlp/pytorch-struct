from .cky import CKY
from .deptree import DepTree
from .linearchain import LinearChain
from .semimarkov import SemiMarkov
from .semirings import (
    LogSemiring,
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
def test_simple(batch, N, C):
    vals = torch.ones(batch, N, C, C)
    semiring = StdSemiring
    alpha = LinearChain(semiring).sum(vals)
    assert (alpha == pow(C, N + 1)).all()
    x = LinearChain(SampledSemiring).marginals(vals)

    x = LinearChain(MultiSampledSemiring).marginals(vals)
    print(MultiSampledSemiring.to_discrete(x, 1))
    print(MultiSampledSemiring.to_discrete(x, 2))


# def test_fb_m():
#     vals = torch.rand(2, 4, 5, 5)
#     v, _, alpha = LinearChain(MaxSemiring)._dp(vals)
#     LinearChain(MaxSemiring)._dp_backward(vals, None, alpha)


# @given(data())
# def test_fb(data):
#     model = data.draw(sampled_from([LinearChain, CKY]))
#     torch.manual_seed(1)
#     vals, (batch, N) = model._rand()

#     lengths = torch.tensor(
#         [data.draw(integers(min_value=2, max_value=N)) for b in range(batch - 1)] + [N]
#     )
#     marginals2 = model().marginals(vals, lengths=lengths, _autograd=True)
#     v, _, alpha = model()._dp(vals, lengths=lengths)
#     marginals = model()._dp_backward(vals, lengths, alpha, v)

#     if isinstance(marginals, tuple):
#         for i, (m1, m2) in enumerate(zip(marginals[:], marginals2[:])):
#             assert torch.isclose(m1, m2).all(), (not torch.isclose(m1, m2)).nonzero()
#     else:
#         assert torch.isclose(marginals, marginals2).all()


@given(data())
def test_entropy(data):
    model = data.draw(sampled_from([LinearChain, SemiMarkov]))
    semiring = EntropySemiring
    struct = model(semiring)
    vals, (batch, N) = model._rand()
    alpha = struct.sum(vals)

    log_z = model(LogSemiring).sum(vals)
    _, log_probs = model(LogSemiring).enumerate(vals)
    log_probs = torch.stack(log_probs, dim=1) - log_z
    print(log_probs.shape, log_z.shape, log_probs.exp().sum(1))
    entropy = -log_probs.mul(log_probs.exp()).sum(1).squeeze(0)
    assert entropy.shape == alpha.shape
    assert torch.isclose(entropy, alpha).all()


@given(data())
@settings(max_examples=50, deadline=None)
def test_generic_a(data):
    model = data.draw(sampled_from([LinearChain, SemiMarkov, CKY, DepTree]))
    semiring = data.draw(sampled_from([LogSemiring, MaxSemiring]))
    struct = model(semiring)
    vals, (batch, N) = model._rand()
    alpha = struct.sum(vals)
    print(alpha.shape)
    count, _ = struct.enumerate(vals)
    print(alpha, count)
    assert alpha.shape[0] == batch
    assert count.shape[0] == batch
    assert alpha.shape == count.shape
    assert torch.isclose(count[0], alpha[0])

    semiring = data.draw(sampled_from([LogSemiring, MaxSemiring, CKY, DepTree]))
    vals, _ = model._rand()
    struct = model(MaxSemiring)
    score = struct.sum(vals)
    marginals = struct.marginals(vals)
    assert torch.isclose(score, struct.score(vals, marginals)).all()


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
    model = data.draw(sampled_from([LinearChain, SemiMarkov, CKY, DepTree]))
    struct = model()
    torch.manual_seed(seed)
    vals, (batch, N) = struct._rand()
    lengths = torch.tensor(
        [data.draw(integers(min_value=2, max_value=N)) for b in range(batch - 1)] + [N]
    )

    m = model(MaxSemiring).marginals(vals, lengths=lengths)
    maxes = struct.score(vals, m)
    part = model().sum(vals, lengths=lengths)
    assert (maxes <= part).all()
    m_part = model(MaxSemiring).sum(vals, lengths=lengths)
    assert (torch.isclose(maxes, m_part)).all(), maxes - m_part

    # m2 = deptree(vals, lengths=lengths)
    # assert (m2 < part).all()

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


@given(data(), integers(min_value=1, max_value=10))
def test_params(data, seed):
    model = data.draw(
        sampled_from([DepTree, CKY])
    )  # LinearChain, SemiMarkov, DepTree, CKY]))
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


def test_hmm():
    C, V, batch, N = 5, 20, 2, 5
    transition = torch.rand(C, C)
    emission = torch.rand(V, C)
    init = torch.rand(C)
    observations = torch.randint(0, V, (batch, N))
    out = LinearChain.hmm(transition, emission, init, observations)
    LinearChain().sum(out)
