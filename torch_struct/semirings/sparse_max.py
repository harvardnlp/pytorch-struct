import torch
from .semirings import _BaseLog


class SparseMaxSemiring(_BaseLog):
    """

    Implements differentiable dynamic programming with a sparsemax semiring (sparsemax, +, -inf, 0).

    Sparse-max gradients give a more sparse set of marginal like terms.

    * From softmax to sparsemax- A sparse model of attention and multi-label classification :cite:`martins2016softmax`
    * Differentiable dynamic programming for structured prediction and attention :cite:`mensch2018differentiable`
    """

    @staticmethod
    def sum(xs, dim=-1):
        return _SimplexProject.apply(xs, dim)


class _SimplexProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, z=1):
        w_star = project_simplex(input, dim)
        ctx.save_for_backward(input, w_star.clone(), torch.tensor(dim))
        x = input.mul(w_star).sum(dim) - w_star.norm(p=2, dim=dim)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        input, w_star, dim = ctx.saved_tensors
        w_star.requires_grad_(True)

        grad_input = None
        if ctx.needs_input_grad[0]:
            wstar = _SparseMaxGrad.apply(w_star, dim)
            grad_input = grad_output.unsqueeze(dim).mul(wstar)
        return grad_input, None, None


class _SparseMaxGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_star, dim):
        ctx.save_for_backward(w_star, dim)
        return w_star

    @staticmethod
    def backward(ctx, grad_output):
        w_star, dim = ctx.saved_tensors
        return sparsemax_grad(grad_output, w_star, dim.item()), None


def project_simplex(v, dim, z=1):
    v_sorted, _ = torch.sort(v, dim=dim, descending=True)
    cssv = torch.cumsum(v_sorted, dim=dim) - z
    ind = torch.arange(1, 1 + v.shape[dim]).to(dtype=v.dtype).to(v.device)
    cond = v_sorted - cssv / ind >= 0
    k = cond.sum(dim=dim, keepdim=True)
    tau = cssv.gather(dim, k - 1) / k.to(dtype=v.dtype)
    w = torch.clamp(v - tau, min=0)
    return w


def sparsemax_grad(dout, w_star, dim):
    out = dout.clone()
    supp = w_star > 0
    out[w_star <= 0] = 0
    nnz = supp.to(dtype=dout.dtype).sum(dim=dim, keepdim=True)
    out = out - (out.sum(dim=dim, keepdim=True) / nnz)
    out[w_star <= 0] = 0
    return out
