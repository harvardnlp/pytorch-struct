from .algorithms import linearchain
import torch

def test_chain():
    linearchain(torch.zeros(10, 20, 5, 5))
