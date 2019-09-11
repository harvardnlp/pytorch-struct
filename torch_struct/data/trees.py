import torchtext.data as data


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
    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                a, b = line.split("\t")
                label = a
                words = [w for w in b.split() if w not in "()"]

                examples.append(data.Example.fromlist((words, label), fields))
        super(ListOpsDataset, self).__init__(examples, fields, **kwargs)
