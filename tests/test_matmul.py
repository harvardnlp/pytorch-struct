import torch
from hypothesis import given
from hypothesis.strategies import integers
import genbmm

bint = integers(min_value=1, max_value=4)
mint = integers(min_value=6, max_value=8)
nint = integers(min_value=3, max_value=5)
kint = integers(min_value=9, max_value=11)


@given(bint, mint, nint, kint)
def test_matmul(batch, m, n, k):
    a, b = torch.rand((m, n)), torch.rand((n, k))
