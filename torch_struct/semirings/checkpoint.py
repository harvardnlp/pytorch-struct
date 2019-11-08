import torch
import numpy as np

def CheckpointSemiring(cls,  max_size, min_size=0):
    class _Check(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            size = [max(p, q) for p, q in zip(a.shape, b.shape)][:-1]
            return accumulate_(a, b, size,
                        lambda a, b: cls.dot(a, b),
                        preserve=len(size),
                        step=max_size // a.shape[-1] + 2)

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            size = [max(p, q) for p, q in zip(a.shape, b.shape)][:-1]
            fn = lambda a, b: cls.dot(a, b)
            grad_a, grad_b = unaccumulate2_(
                a, b, grad_output, len(grad_output.shape), fn,
                step=max_size // a.shape[-1] + 2
            )
            return grad_a, grad_b

    class _CheckpointSemiring(cls):
        @staticmethod
        def dot(a, b):
            size = torch.tensor([max(i,j) for i, j in zip(a.shape, b.shape)]).prod()
            if size < min_size:
                return cls.dot(a, b)
            else:
                return _Check.apply(a, b)

    return _CheckpointSemiring


def ones(x):
    one = []
    for i, v in enumerate(x.shape[:-1]):
        if v == 1:
            one.append(i)
    return one

def mind(one, inds):
    inds = list(inds)
    for v in one:
        inds[v] = inds[v].clone().fill_(0)
    return inds

def accumulate_(a, b, size, fn, preserve, step=10000):
    slices = []
    total = 1
    for s in size[:preserve]:
        slices.append(slice(s))
        total *= s
    if step > total:
        return fn(a, b)

    ret = torch.zeros(*size, dtype=a.dtype, device=a.device)
    a_one, b_one = ones(a), ones(b)
    indices = torch.tensor(np.mgrid[slices]).view(len(ret.shape[:preserve]), -1)

    for p in range(0, total, step):
        ind = indices[:, p : p + step].unbind()
        a_ind = mind(a_one, ind)
        b_ind = mind(b_one, ind)
        ret[ind] = fn(a[tuple(a_ind)], b[tuple(b_ind)])
    return ret

# def unaccumulate_(a, b, grad_output, fn, step=10000):
#     slices = []
#     a_grad = a.clone().fill_(0)
#     b_grad = b.clone().fill_(0)

#     total = 1
#     for s in grad_output.shape:
#         slices.append(slice(s))
#         total *= s
#     a_one, b_one = ones(a), ones(b)

#     indices = torch.tensor(np.mgrid[slices]).view(len(grad_output.shape), -1)

#     for p in range(0, total, step):
#         ind = indices[:, p : p + step].unbind()
#         a_ind = mind(a_one, ind)
#         b_ind = mind(b_one, ind)

#         q = fn(a[tuple(a_ind)], b[tuple(b_ind)], grad_output[tuple(ind)])
#         a_grad.index_put_(tuple(a_ind),  q, accumulate=True)
#         b_grad.index_put_(tuple(b_ind),  q, accumulate=True)
#     return a_grad, b_grad

def unaccumulate2_(a, b, grad_output, preserve, fn, step=10000):
    slices = []
    a_grad = a.clone().fill_(0)
    b_grad = b.clone().fill_(0)

    total = 1
    for s in grad_output.shape[:preserve]:
        slices.append(slice(s))
        total *= s

    if step > total:
        with torch.enable_grad():
            a_in = a.clone().requires_grad_(True)
            b_in = b.clone().requires_grad_(True)
            q = fn(a, b)
        ag, bg = torch.autograd.grad(q, (a, b), grad_output)
        return ag, bg

    a_one, b_one = ones(a), ones(b)
    print(a.shape, b.shape, a_one, b_one, preserve)
    indices = torch.tensor(np.mgrid[slices]).view(len(grad_output.shape[:preserve]), -1)

    for p in range(0, total, step):
        ind = indices[:, p : p + step].unbind()
        a_ind = mind(a_one, ind)
        b_ind = mind(b_one, ind)

        with torch.enable_grad():
            a_in = a.clone().requires_grad_(True)
            b_in = b.clone().requires_grad_(True)
            q = fn(a[tuple(a_ind)], b[tuple(b_ind)])
        ag, bg = torch.autograd.grad(q, (a, b), grad_output[tuple(ind)])
        a_grad += ag
        b_grad += bg

    return a_grad, b_grad
