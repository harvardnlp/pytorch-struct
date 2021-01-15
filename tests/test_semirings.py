import torch
from hypothesis import given
from hypothesis.strategies import integers


from torch_struct import (
    LogSemiring,
    CheckpointSemiring,
    CheckpointShardSemiring,
    KMaxSemiring,
    MaxSemiring,
    StdSemiring,
)


lint = integers(min_value=2, max_value=10)


@given(lint, lint, lint)
def test_max(a, b, c):
    torch.manual_seed(0)
    t1 = torch.rand(a, 1, c).requires_grad_(True)
    t2 = torch.rand(1, b, c).requires_grad_(True)
    r1 = MaxSemiring.dot(t1, t2)

    t1a = torch.zeros(2, a, 1, c)
    t2a = torch.zeros(2, 1, b, c)
    t1a[0] = t1
    t2a[0] = t2
    t1a[1].fill_(-1e10)
    t2a[1].fill_(-1e10)

    r2 = KMaxSemiring(2).dot(t1a, t2a)
    assert torch.isclose(r1, r2[0]).all()

    (a, b) = torch.autograd.grad(r1.sum(), (t1, t2))
    (a2, b2) = torch.autograd.grad(r2[0].sum(), (t1a, t2a))

    assert torch.isclose(a, a2[0]).all()
    assert torch.isclose(b, b2[0]).all()


@given(lint, lint, lint)
def test_checkpoint(a, b, c):
    torch.manual_seed(0)
    t1 = torch.rand(a, 1, c).requires_grad_(True)
    t2 = torch.rand(1, b, c).requires_grad_(True)

    r1 = LogSemiring.dot(t1, t2)
    r2 = CheckpointSemiring(LogSemiring).dot(t1, t2)
    r2 = CheckpointShardSemiring(LogSemiring, 2).dot(t2, t1)
    assert torch.isclose(r1, r2).all()

    (a1, b1) = torch.autograd.grad(r1.sum(), (t1, t2))
    (a2, b2) = torch.autograd.grad(r2.sum(), (t1, t2))

    assert torch.isclose(a1, a2).all()
    assert torch.isclose(b1, b2).all()


@given(lint, lint, lint, lint)
def test_matmul(a, b, c, d):
    torch.manual_seed(0)
    t1 = torch.rand(a, b, c).requires_grad_(True)
    t2 = torch.rand(a, c, d).requires_grad_(True)

    r1 = StdSemiring.matmul(t1, t2)
    r2 = StdSemiring.sum(
        StdSemiring.times(
            t1.unsqueeze(-2).view(a, b, 1, c),
            t2.transpose(-2, -1).unsqueeze(-3).view(a, 1, d, c),
        )
    )
    print(r1.shape, r2.shape, a, b, d)
    assert torch.isclose(r1, r2).all()

    # (a1, b1) = torch.autograd.grad(r1.sum(), (t1, t2))
    # (a2, b2) = torch.autograd.grad(r2.sum(), (t1, t2))

    # assert torch.isclose(a1, a2).all()
    # assert torch.isclose(b1, b2).all()
