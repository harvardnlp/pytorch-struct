import torch

has_genbmm = False
try:
    import genbmm
    from genbmm import BandedMatrix

    has_genbmm = True
except ImportError:
    pass


def broadcast_size(a, b):
    return torch.tensor([max(i, j) for i, j in zip(a.shape, b.shape)]).prod()


def matmul_size(a, b):
    size = [max(i, j) for i, j in zip(a.shape[:-2], b.shape[:-2])]
    size.append(a.shape[-2])
    size.append(b.shape[-1])
    return size


def CheckpointSemiring(cls, min_size=0):
    class _Check(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return cls.matmul(a, b)

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            with torch.enable_grad():
                q = cls.matmul(a, b)
                return torch.autograd.grad(q, (a, b), grad_output)

    class _CheckBand(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, a_lu, a_ld, b, b_lu, b_ld):
            ctx.save_for_backward(a, b, torch.LongTensor([a_lu, a_ld, b_lu, b_ld]))
            a = BandedMatrix(a, a_lu, a_ld)
            b = BandedMatrix(b, b_lu, b_ld)
            return cls.matmul(a, b).data

        @staticmethod
        def backward(ctx, grad_output):
            a, b, bands = ctx.saved_tensors
            a_lu, a_ld, b_lu, b_ld = bands.tolist()
            with torch.enable_grad():
                q = cls.matmul(BandedMatrix(a, a_lu, a_ld), BandedMatrix(b, b_lu, b_ld))
                grad_a, grad_b = torch.autograd.grad(q.data, (a, b), grad_output)
                return grad_a, None, None, grad_b, None, None

    class _CheckpointSemiring(cls):
        @staticmethod
        def matmul(a, b):
            if has_genbmm and isinstance(a, genbmm.BandedMatrix):
                lu = a.lu + b.lu
                ld = a.ld + b.ld
                c = _CheckBand.apply(a.data, a.lu, a.ld, b.data, b.lu, b.ld)
                return BandedMatrix(c, lu, ld, cls.zero)

            if broadcast_size(a, b) > min_size:
                return _Check.apply(a, b)
            else:
                return cls.matmul(a, b)

    return _CheckpointSemiring


def CheckpointShardSemiring(cls, max_size, min_size=0):
    class _Check(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            size = matmul_size(a, b)
            return accumulate_(
                a,
                b,
                size,
                lambda a, b: cls.matmul(a, b),
                preserve=len(size),
                step=max_size // (b.shape[-2] * a.shape[-1]) + 2,
            )

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            grad_a, grad_b = unaccumulate_(
                a,
                b,
                grad_output,
                len(grad_output.shape),
                lambda a, b: cls.matmul(a, b),
                step=max_size // (b.shape[-2] * a.shape[-1]) + 2,
            )
            return grad_a, grad_b

    class _CheckpointSemiring(cls):
        @staticmethod
        def matmul(a, b):
            size = torch.tensor([max(i, j) for i, j in zip(a.shape, b.shape)]).prod()
            if size < min_size:
                return cls.matmul(a, b)
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

    a = a.expand(*size[:-2], a.shape[-2], a.shape[-1])
    b = b.expand(*size[:-2], b.shape[-2], b.shape[-1])

    a2 = a.contiguous().view(-1, a.shape[-2], a.shape[-1])
    b2 = b.contiguous().view(-1, b.shape[-2], b.shape[-1])
    ret = ret.view(-1, a.shape[-2], b.shape[-1])
    for p in range(0, ret.shape[0], step):
        ret[p : p + step, :] = fn(a2[p : p + step], b2[p : p + step])
    ret = ret.view(*size)
    return ret


def unaccumulate_(a, b, grad_output, preserve, fn, step=10000):
    slices = []
    total = 1
    size = grad_output.shape[:preserve]
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

    a2 = a.expand(*size[:-2], a.shape[-2], a.shape[-1])
    b2 = b.expand(*size[:-2], b.shape[-2], b.shape[-1])
    a2 = a2.contiguous().view(-1, a.shape[-2], a.shape[-1])
    b2 = b2.contiguous().view(-1, b.shape[-2], b.shape[-1])

    a_grad = a2.clone().fill_(0)
    b_grad = b2.clone().fill_(0)

    grad_output = grad_output.view(-1, a.shape[-2], b.shape[-1])
    for p in range(0, grad_output.shape[0], step):
        with torch.enable_grad():
            a_in = a2[p : p + step].clone().requires_grad_(True)
            b_in = b2[p : p + step].clone().requires_grad_(True)
            q = fn(a_in, b_in)
        ag, bg = torch.autograd.grad(q, (a_in, b_in), grad_output[p : p + step])
        a_grad[p : p + step] += ag
        b_grad[p : p + step] += bg

    a_grad = a_grad.view(*size[:-2], a.shape[-2], a.shape[-1])
    b_grad = b_grad.view(*size[:-2], b.shape[-2], b.shape[-1])
    a_ones = ones(a)
    b_ones = ones(b)
    f1, f2 = a_grad.sum(a_ones, keepdim=True), b_grad.sum(b_ones, keepdim=True)
    return f1, f2


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
