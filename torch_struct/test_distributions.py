from .distributions import LinearChainCRF
from .autoregressive import Autoregressive
import torch
from hypothesis import given, settings
from hypothesis.strategies import integers, data, sampled_from

smint = integers(min_value=2, max_value=4)
tint = integers(min_value=1, max_value=2)
lint = integers(min_value=2, max_value=10)


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
    edges, enum_lengths = dist.enumerate_support()
    print(edges.shape)
    log_probs = dist.log_prob(edges)
    for b in range(lengths.shape[0]):
        log_probs[enum_lengths[b] :, b] = -1e9

    assert torch.isclose(log_probs.exp().sum(0), torch.tensor(1.0)).all()

    entropy = dist.entropy
    assert torch.isclose(entropy, -log_probs.exp().mul(log_probs).sum(0)).all()

    argmax = dist.argmax
    _, max_indices = log_probs.max(0)

    amax = edges[max_indices, torch.arange(batch)]
    print(argmax.nonzero())
    print((amax - argmax).nonzero(), lengths)
    assert (amax == argmax).all()

    samples = dist.sample((100,))
    marginals = dist.marginals
    assert ((samples.mean(0) - marginals).abs() < 0.2).all()


@given(data(), integers(min_value=1, max_value=20))
@settings(max_examples=50, deadline=None)
def test_autoregressive(data, seed):
    n_classes = 2
    n_length = 5
    batch = 3

    values = torch.rand(batch, n_length, n_classes)

    values2 = values.unsqueeze(-1).expand(batch, n_length, n_classes, n_classes).clone()
    values2[:, 0, :, :] = -1e9
    values2[:, 0, torch.arange(n_classes), torch.arange(n_classes)] = values[:, 0]

    init = (torch.zeros(batch, 5).long(),)

    class Model(torch.nn.Module):
        def forward(self, inputs, state):
            if inputs.shape[1] == 1:
                state, = state
                in_batch, hidden = state.shape
                t = state[0, 0]
                batch = values.shape[0]
                x = values[:, t, :].unsqueeze(0).expand(in_batch//batch, batch, n_classes)
                state = (state + 1,)
                return x.contiguous().view(in_batch, n_classes), state
            else:
                return values, state

    auto = Autoregressive(Model(), init, n_classes, n_length)
    v = auto.greedy_argmax()
    batch, n, c = v.shape
    assert(n == n_length)
    assert(c == n_classes)

    assert (v == LinearChainCRF(values2).argmax.sum(-1)).all()
    crf = LinearChainCRF(values2)
    v2 = auto.beam_topk(K=5)

    assert (v2.nonzero() == crf.topk(5).sum(-1).nonzero()).all()
    assert (v2[0] == LinearChainCRF(values2).argmax.sum(-1)).all()

    print(auto.log_prob(v, normalize=False))
    print(crf.struct().score(crf.argmax, values2))
    assert (
        auto.log_prob(v, normalize=False) == crf.struct().score(crf.argmax, values2)
    ).all()

    assert auto.sample((7,)).shape == (7, batch, n_length, n_classes)

    assert auto.sample_without_replacement((7,)).shape == (
        7,
        batch,
        n_length,
        n_classes,
    )


def test_ar2():
    batch, N, C, H = 3, 10, 2, 5
    layer = 1

    def t(a):
        return tuple((t.transpose(0, 1) for t in a))

    init = (torch.zeros(batch, layer, H),)
    class AR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.RNN(H, H, batch_first=True)
            self.proj = torch.nn.Linear(H, C)
            self.embed = torch.nn.Embedding(C, H)

        def forward(self, inputs, state):
            inputs = self.embed(inputs)
            out, state = self.rnn(inputs,
                                  t(state)[0])
            out = self.proj(out)
            return out, t((state,))

    dist = Autoregressive(AR(), init, C, N)
    dist.greedy_argmax()
    dist.beam_topk(7)

    lstm = torch.nn.LSTM(H, H)
    proj = torch.nn.Linear(H, C)
    embed = torch.nn.Embedding(C, H)

    init = (torch.zeros(batch, layer, H),
            torch.zeros(batch, layer, H))
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
    dist.greedy_argmax()
    dist.beam_topk(5)
