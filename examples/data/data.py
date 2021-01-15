import torchtext
import torch


def token_pre(tokenizer, q):
    st = " ".join(q)
    s = tokenizer.tokenize(st)

    out = [0]
    cur = 0
    expect = ""
    first = True
    for i, w in enumerate(s):
        if len(expect) == 0:
            cur += 1
            expect = q[cur - 1].lower()
            first = True
        if w.startswith("##"):
            out.append(-1)
            expect = expect[len(w) - 2 :]
        elif first:
            out.append(cur)
            expect = expect[len(w) :]
            first = False
        else:
            expect = expect[len(w) :]
    out.append(cur + 1)
    # assert cur == len(q)-1, "%s %s \n%s\n%s"%(len(q), cur, q, s)
    if cur != len(q):
        print("error")
        return [0] * (len(q) + 2), [0] * (len(q) + 2)
    return tokenizer.encode(st, add_special_tokens=True), out


def token_post(ls):
    lengths = [len(l[0]) for l in ls]

    length = max(lengths)
    out = [l[0] + ([0] * (length - len(l[0]))) for l in ls]

    lengths2 = [max(l[1]) + 1 for l in ls]
    length2 = max(lengths2)
    out2 = torch.zeros(len(ls), length, length2)
    for b, l in enumerate(ls):
        for i, w in enumerate(l[1]):
            if w != -1:
                out2[b, i, w] = 1
    return torch.LongTensor(out), out2.long(), lengths


def SubTokenizedField(tokenizer):
    """
    Field for use with pytorch-transformer
    """
    FIELD = torchtext.data.RawField(
        preprocessing=lambda s: token_pre(tokenizer, s), postprocessing=token_post
    )
    FIELD.is_target = False
    return FIELD


def TokenBucket(
    train, batch_size, device="cuda:0", key=lambda x: max(len(x.word[0]), 5)
):
    def batch_size_fn(x, _, size):
        return size + key(x)

    return torchtext.data.BucketIterator(
        train,
        train=True,
        sort=False,
        sort_within_batch=True,
        shuffle=True,
        batch_size=batch_size,
        sort_key=lambda x: key(x),
        repeat=True,
        batch_size_fn=batch_size_fn,
        device=device,
    )
