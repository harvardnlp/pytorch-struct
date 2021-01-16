from torch_struct import LinearChainCRF, Autoregressive, KMaxSemiring, LogSemiring
import torch
from hypothesis import given, settings
from hypothesis.strategies import integers, data, sampled_from
from .extensions import test_lookup

smint = integers(min_value=2, max_value=4)
tint = integers(min_value=1, max_value=2)
lint = integers(min_value=2, max_value=10)


def enumerate_support(dist):
    """
    Compute the full exponential enumeration set.

    Parameters:
        dist : Distribution

    Returns:
        (enum, enum_lengths) - (*tuple cardinality x batch_shape x event_shape*)
    """
    _, _, edges, enum_lengths = test_lookup[dist.struct]().enumerate(
        LogSemiring, dist.log_potentials, dist.lengths
    )
    # if expand:
    #     edges = edges.unsqueeze(1).expand(edges.shape[:1] + self.batch_shape[:1] + edges.shape[1:])
    return edges, enum_lengths


@given(data(), integers(min_value=1, max_value=20))
@settings(max_examples=50, deadline=None)
def test_simple(data, seed):

    model = data.draw(sampled_from([LinearChainCRF]))
    struct = model.struct
    torch.manual_seed(seed)
    vals, (batch, N) = struct._rand()
    lengths = torch.tensor(
        [data.draw(integers(min_value=2, max_value=N)) for b in range(batch - 1)] + [N]
    )
    dist = model(vals, lengths)
    edges, enum_lengths = enumerate_support(dist)
    log_probs = dist.log_prob(edges)
    for b in range(lengths.shape[0]):
        log_probs[enum_lengths[b] :, b] = -1e9
    assert torch.isclose(log_probs.exp().sum(0), torch.tensor(1.0)).all()
    entropy = dist.entropy
    assert torch.isclose(entropy, -log_probs.exp().mul(log_probs).sum(0)).all()

    vals2 = torch.rand(*vals.shape)
    dist2 = model(vals2, lengths)

    cross_entropy = dist.cross_entropy(other=dist2)
    kl = dist.kl(other=dist2)

    edges2, enum_lengths2 = enumerate_support(dist2)
    log_probs2 = dist2.log_prob(edges2)
    for b in range(lengths.shape[0]):
        log_probs2[enum_lengths2[b] :, b] = -1e9

    assert torch.isclose(cross_entropy, -log_probs.exp().mul(log_probs2).sum(0)).all()
    assert torch.isclose(kl, -log_probs.exp().mul(log_probs2 - log_probs).sum(0)).all()

    argmax = dist.argmax
    _, max_indices = log_probs.max(0)

    amax = edges[max_indices, torch.arange(batch)]
    print(argmax.nonzero())
    print((amax - argmax).nonzero(), lengths)
    assert (amax == argmax).all()

    samples = dist.sample((100,))
    marginals = dist.marginals
    assert ((samples.mean(0) - marginals).abs() < 0.2).all()

    dist.kmax(5)
    dist.count


@given(data(), integers(min_value=1, max_value=20))
@settings(max_examples=50, deadline=None)
def test_autoregressive(data, seed):
    n_classes = 2
    n_length = 5
    batch = 3
    BATCH = 3

    values = torch.rand(batch, n_length, n_classes)

    values2 = values.unsqueeze(-1).expand(batch, n_length, n_classes, n_classes).clone()
    values2[:, 0, :, :] = -1e9
    values2[:, 0, torch.arange(n_classes), torch.arange(n_classes)] = values[:, 0]

    init = (torch.zeros(batch, 5).long(),)

    class Model(torch.nn.Module):
        def forward(self, inputs, state):
            if inputs.shape[1] == 1:
                (state,) = state
                in_batch, hidden = state.shape
                t = state[0, 0]
                batch = values.shape[0]
                x = (
                    values[:, t, :]
                    .unsqueeze(0)
                    .expand(in_batch // batch, batch, n_classes)
                )
                state = (state + 1,)
                return x.contiguous().view(in_batch, 1, n_classes), state
            else:
                return (
                    torch.cat((values, torch.zeros(BATCH, 1, n_classes)), dim=1),
                    state,
                )

    auto = Autoregressive(Model(), init, n_classes, n_length, normalize=False)
    v, _, _ = auto.greedy_max()
    batch, n, c = v.shape
    assert n == n_length
    assert c == n_classes

    assert (v == LinearChainCRF(values2).argmax.sum(-1)).all()
    crf = LinearChainCRF(values2)
    v2 = auto.beam_topk(K=5)

    assert (v2.nonzero() == crf.topk(5).sum(-1).nonzero()).all()
    assert (v2[0] == LinearChainCRF(values2).argmax.sum(-1)).all()

    print(auto.log_prob(v.unsqueeze(0)))
    print(crf.struct().score(crf.argmax, values2))
    assert (
        torch.isclose(
            auto.log_prob(v.unsqueeze(0)), crf.struct().score(crf.argmax, values2)
        )
    ).all()
    assert auto.sample((7,)).shape == (7, batch, n_length, n_classes)

    assert auto.sample_without_replacement((7,)).shape == (
        7,
        batch,
        n_length,
        n_classes,
    )


def test_ar2():
    batch, N, C, H = 3, 5, 2, 5
    layer = 1

    def t(a):
        return tuple((t.transpose(0, 1) for t in a))

    init = (torch.zeros(batch, layer, H),)

    class AR(torch.nn.Module):
        def __init__(self, sparse=True):
            super().__init__()
            self.sparse = sparse
            self.rnn = torch.nn.RNN(H, H, batch_first=True)
            self.proj = torch.nn.Linear(H, C)
            if sparse:
                self.embed = torch.nn.Embedding(C, H)
            else:
                self.embed = torch.nn.Linear(C, H)

        def forward(self, inputs, state):
            if not self.sparse and inputs.dim() == 2:
                inputs = torch.nn.functional.one_hot(inputs, C).float()
            inputs = self.embed(inputs)
            out, state = self.rnn(inputs, t(state)[0])
            out = self.proj(out)
            return out, t((state,))

    dist2 = Autoregressive(AR(sparse=False), init, C, N, normalize=False)
    path, _, _ = dist2.greedy_tempmax(1)

    dist = Autoregressive(AR(), init, C, N, normalize=False)

    path, scores, _ = dist.greedy_max()

    assert torch.isclose(scores, dist.log_prob(path.unsqueeze(0))).all()
    scores = dist._beam_max(7)
    path = dist.beam_topk(7)

    a = torch.tensor([[1, 2, 5], [3, 4, 6]])
    print(KMaxSemiring(2).sparse_sum(a))

    import itertools

    for i in range(5):
        print(dist.log_prob(path)[i])
        print(scores[i])
        print(path[i, 0])

    print(path.shape, scores.shape)
    assert torch.isclose(scores, dist.log_prob(path)).all()

    v = torch.tensor(list(itertools.product([0, 1], repeat=N)))

    v = v.unsqueeze(1).expand(v.shape[0], batch, N)
    all_scores = dist.log_prob(v, sparse=True)
    best, ind = torch.max(all_scores, dim=0)
    assert torch.isclose(scores[0, 0], best[0]).all()

    print(v[ind[0], 0].shape, path[0, 0].shape)
    assert (torch.nn.functional.one_hot(v[ind, 0], C) == path[0, 0].long()).all()

    dist = Autoregressive(AR(), init, C, N, normalize=True)
    path = dist.sample((7,))

    init = (torch.zeros(batch, layer, H), torch.zeros(batch, layer, H))

    class AR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(H, H, batch_first=True)
            self.proj = torch.nn.Linear(H, C)
            self.embed = torch.nn.Embedding(C, H)

        def forward(self, inputs, state):
            inputs = self.embed(inputs)
            out, state = self.rnn(inputs, t(state))
            out = self.proj(out)
            return out, t(state)

    dist = Autoregressive(AR(), init, C, N)
    dist.greedy_max()
    dist.beam_topk(5)
