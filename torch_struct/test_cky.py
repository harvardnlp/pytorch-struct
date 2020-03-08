import torch

from torch_struct import SentCFG


def params_l3():
    """
    seq = x y z, t0, t1 & n0, n1, n2
    """
    terms = [[2, 1], [1, 2], [1, 1]]
    # term4 = [[1, 1], [2, 1], [1, 2]]
    roots = [1, 1, 1]
    rule1 = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 5],
        [1, 1, 1, 2, 1],
    ]
    rule2 = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 6],
        [1, 1, 1, 2, 1],
    ]
    rule3 = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 4, 5],
        [1, 1, 1, 8, 9],
        [1, 1, 1, 1, 4],
        [1, 1, 1, 2, 1],
    ]
    rules = [rule1, rule2, rule3]

    terms = (
        torch.tensor(terms, dtype=torch.float64, requires_grad=True)
        .unsqueeze(0)
        .float()
    )
    roots = (
        torch.tensor(roots, dtype=torch.float64, requires_grad=True)
        .unsqueeze(0)
        .float()
    )
    rules = (
        torch.tensor(rules, dtype=torch.float64, requires_grad=True)
        .unsqueeze(0)
        .float()
    )

    length = torch.tensor([3]).long()

    # print('term:\n', terms, terms.shape)
    # print('root:\n', roots, roots.shape)
    # print('rule:\n', rules, rules.shape)
    return ((terms, rules, roots), length)


def extract_parse(span, length):
    tree = [(i, str(i)) for i in range(length)]
    tree = dict(tree)
    spans = []
    cover = (span > 0).float().nonzero()
    for i in range(cover.shape[0]):
        w, r, A = cover[i].tolist()
        w = w + 1
        r = r + w
        l = r - w
        spans.append((l, r, A))
        span = "({} {})".format(tree[l], tree[r])
        tree[r] = tree[l] = span
    return spans, tree[0]


def extract_topk(matrix, lengths):
    batch, K, N = matrix.shape[:3]
    spans = []
    trees = []
    for b in range(batch):
        for k in range(K):
            this_span = matrix[b][k]
            span, tree = extract_parse(this_span, lengths[b])
            trees.append(tree)
            spans.append(span)
            # print(span)
            # print(tree)
        # break
    return spans, trees


def extract_parses(matrix, lengths):
    batch, K, N = matrix.shape[:3]
    spans = []
    trees = []
    for b in range(batch):
        span, tree = extract_parse(matrix[b], lengths[b])
        trees.append(tree)
        spans.append(span)
        # print(span, tree)
        # break
    return spans, trees


def test_l3_kbest():
    params, lengths = params_l3()
    dist = SentCFG(params, lengths=lengths)

    _, _, _, spans = dist.argmax
    spans, trees = extract_parses(spans, lengths)
    best_trees = "((0 1) 2)"
    best_spans = [(0, 1, 2), (0, 2, 2)]
    assert spans[0] == best_spans
    assert trees[0] == best_trees

    _, _, _, spans = dist.topk(4)
    size = (1, 0) + tuple(range(2, spans.dim()))
    spans = spans.permute(size)
    spans, trees = extract_topk(spans, lengths)
    best_trees = "((0 1) 2)"
    best_spans = [
        [(0, 1, 2), (0, 2, 2)],
        [(0, 1, 2), (0, 2, 2)],
        [(0, 1, 1), (0, 2, 2)],
        [(0, 1, 1), (0, 2, 2)],
    ]
    for i, (span, tree) in enumerate(zip(spans, trees)):
        assert span == best_spans[i]
        assert tree == best_trees


if __name__ == "__main__":
    test_l3_kbest()
