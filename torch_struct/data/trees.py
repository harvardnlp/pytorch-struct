import torchtext.data as data
import torch


class ConllXDataset(data.Dataset):
    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        columns = [[], []]
        column_map = {1: 0, 6: 1}
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = [[], []]
                else:
                    for i, column in enumerate(line.split(separator)):
                        if i in column_map:
                            columns[column_map[i]].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(ConllXDataset, self).__init__(examples, fields, **kwargs)


class ListOpsDataset(data.Dataset):
    @staticmethod
    def tree_field(v):
        def post(ls):
            batch = len(ls)
            lengths = [l[-1][1] for l in ls]
            length = max(lengths) + 1
            ret = torch.zeros(batch, length, length, 10 + len(v.vocab))
            for b in range(len(ls)):
                for i, j, n in ls[b]:
                    if i == j:
                        ret[b, i, j, v.vocab.stoi[n] + 1] = 1
                    else:
                        ret[b, i, j, 0] = 1
            return ret.long()

        return post

    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                a, b = line.split("\t")
                label = a
                words = [w for w in b.split() if w not in "()"]

                cur = 0
                spans = []
                stack = []
                for w in b.split():
                    if w == "(":
                        stack.append(cur)
                    elif w == ")":
                        spans.append((stack[-1], cur - 1, w))
                        stack = stack[:-1]
                    else:
                        spans.append((cur, cur, w))
                        cur += 1
                examples.append(data.Example.fromlist((words, label, spans), fields))
        super(ListOpsDataset, self).__init__(examples, fields, **kwargs)
