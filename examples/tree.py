# -*- coding: utf-8 -*-
# wandb login 7cd7ade39e2d850ec1cf4e914d9a148586a20900
from torch_struct import TreeCRF, SelfCritical
import torchtext.data as data
from torch_struct.data import ListOpsDataset, TokenBucket
from torch_struct.networks import TreeLSTM, SpanLSTM
import torch
import torch.nn as nn
import wandb


config = {
    "method": "reinforce",
    "baseline": "mean",
    "opt": "adadelta",
    "lr_struct": 0.1,
    "lr_params": 1,
    "train_model": True,
    "var_norm": False,
    "entropy": 0.001,
    "v": 3,
    "RL_K": 5,
    "H": 100,
    "train_len": 100,
    "div_ent": 1,
}


NAME = "yoyo3"

wandb.init(project="pytorch-struct", config=config)


def clip(p):
    torch.nn.utils.clip_grad_norm_(parameters=p, max_norm=0.5, norm_type=float("inf"))


def expand_spans(spans, words, K, V):
    batch, N = words.shape
    spans[:, torch.arange(N), torch.arange(N)] = 0
    new_spans = torch.zeros(K, batch, N, N, 1 + V).cuda()
    new_spans[:, :, :, :, :1] = spans.view(K, batch, N, N, 1)
    new_spans[:, :, torch.arange(N), torch.arange(N), :].fill_(0)
    new_spans[:, :, torch.arange(N), torch.arange(N), 1:] = (
        torch.nn.functional.one_hot(words, V).float().cuda().view(1, batch, N, V)
    )
    new_spans = new_spans.view(batch * K, N, N, 1 + V)
    return new_spans


def valid_sup(valid_iter, model, tree_lstm, V):
    total = 0
    correct = 0
    Dist = TreeCRF
    for i, ex in enumerate(valid_iter):
        words, lengths = ex.word
        trees = ex.tree
        label = ex.label
        words = words.cuda()

        def tree_reward(spans):
            new_spans = expand_spans(spans, words, 1, V)
            g, labels, indices, topo = TreeLSTM.spans_to_dgl(new_spans)
            _, am = tree_lstm(g, labels, indices, topo, lengths).max(-1)
            return (label == am).sum(), label.shape[0]

        words = words.cuda()
        phi = model(words, lengths)
        dist = TreeCRF(phi, lengths)
        argmax = dist.argmax
        argmax_tree = dist.struct.from_parts(argmax.detach())[0]
        score, tota = tree_reward(argmax_tree)
        total += int(tota)
        correct += score

        if i == 25:
            break
    print(correct.item() / float(total), correct, total)
    return correct.item() / float(total)


def run_train(train_iter, valid_iter, model, tree_lstm, V):
    opt_struct = torch.optim.Adadelta(list(model.parameters()), lr=config["lr_struct"])
    opt_params = torch.optim.Adadelta(
        list(tree_lstm.parameters()), lr=config["lr_params"]
    )

    model.train()
    tree_lstm.train()
    losses = []
    Dist = TreeCRF
    step = 0
    trees = None
    for epoch in range(100):
        print("Epoch", epoch)

        for i, ex in enumerate(train_iter):
            step += 1
            words, lengths = ex.word
            label = ex.label
            batch = label.shape[0]
            _, N = words.shape
            words = words.cuda()

            def tree_reward(spans, K):
                new_spans = expand_spans(spans, words, K, V)
                g, labels, indices, topo = TreeLSTM.spans_to_dgl(new_spans)
                ret = tree_lstm(
                    g, labels, indices, topo, torch.cat([lengths for _ in range(K)])
                )
                ret = ret.view(K, batch, -1)
                return -ret[:, torch.arange(batch), label].view(K, batch)

            sc = SelfCritical(tree_reward)
            phi = model(words, lengths)
            dist = Dist(phi)
            structs, rewards, score, max_score = sc.forward(dist, K=config["RL_K"])

            if config["train_model"]:
                opt_params.zero_grad()
                score.mean().backward()
                clip(tree_lstm.parameters())
                opt_params.step()
                opt_params.zero_grad()

            if config["method"] == "reinforce":
                opt_struct.zero_grad()
                entropy = dist.entropy
                r = dist.log_prob(structs)
                obj = rewards.mul(r).mean(-1).mean(-1)
                policy = (
                    obj - config["entropy"] * entropy.div(lengths.float().cuda()).mean()
                )
                policy.backward()
                clip(model.parameters())
                opt_struct.step()
            losses.append(-max_score.mean().detach())

            # DEBUG
            if i % 50 == 9:
                print(torch.tensor(losses).mean(), words.shape)
                print("Round")
                print("Entropy", entropy.mean().item())
                print("Reward", rewards.mean().item())
                if i % 1000 == 9:
                    valid_loss = valid_sup(valid_iter, model, tree_lstm, V)
                    fname = "/tmp/checkpoint.%s.%0d.%0d.%s" % (
                        NAME,
                        epoch,
                        i,
                        valid_loss,
                    )
                    torch.save((model, tree_lstm), fname)
                    wandb.save(fname)
                    trees = valid_show(valid_iter, model)
                else:
                    print(valid_loss)

                wandb.log(
                    {
                        "entropy": entropy.mean(),
                        "valid_loss": valid_loss,
                        "reward": rewards.mean(),
                        "step": step,
                        "tree": trees,
                        "reward_var": rewards.var(),
                        "loss": torch.tensor(losses).mean(),
                    }
                )
                losses = []


def valid_show(valid_iter, model):
    table = wandb.Table(columns=["Sent", "Predicted Tree", "True Tree"])
    Dist = TreeCRF
    for i, ex in enumerate(valid_iter):
        words, lengths = ex.word
        label = ex.label
        batch = label.shape[0]
        words = words.cuda()
        phi = model(words, lengths)
        dist = Dist(phi)
        argmax = dist.argmax
        argmax_tree = dist.struct.from_parts(argmax.detach())[0].cpu()
        for b in range(words.shape[0]):
            out = [WORD.vocab.itos[w.item()] for w in words[b]]
            sent = " ".join(out)

            def show(tree):
                output = ""
                start = {}
                end = {}
                for i, j, _ in tree.nonzero():
                    i = i.item()
                    j = j.item()
                    start.setdefault(i, -1)
                    end.setdefault(j, -1)
                    start[i] += 1
                    end[j] += 1
                for i, w in enumerate(out):
                    for _ in range(start.get(i, 0)):
                        output += "( "
                    output += w + " "
                    for _ in range(end.get(i, 0)):
                        output += ") "
                return output

            predict_text = show(ex.tree[b].cpu())
            true_text = show(argmax_tree[b].cpu())
            table.add_data(sent, predict_text, true_text)
        break
    return table


WORD = None


def main():
    global WORD
    WORD = data.Field(
        include_lengths=True, batch_first=True, eos_token=None, init_token=None
    )
    LABEL = data.Field(sequential=False, batch_first=True)
    TREE = data.RawField(postprocessing=ListOpsDataset.tree_field(WORD))
    TREE.is_target = False
    train = ListOpsDataset(
        "data/train_d20s.tsv",
        (("word", WORD), ("label", LABEL), ("tree", TREE)),
        filter_pred=lambda x: 5 < len(x.word) < config["train_len"],
    )
    WORD.build_vocab(train)
    LABEL.build_vocab(train)
    valid = ListOpsDataset(
        "data/test_d20s.tsv",
        (("word", WORD), ("label", LABEL), ("tree", TREE)),
        filter_pred=lambda x: 5 < len(x.word) < 150,
    )

    train_iter = TokenBucket(
        train, batch_size=1500, device="cuda:0", key=lambda x: len(x.word)
    )
    train_iter.repeat = False
    valid_iter = data.BucketIterator(
        train, batch_size=50, train=False, sort=False, device="cuda:0"
    )

    NT = 1
    T = len(WORD.vocab)
    V = T

    if True:
        tree_lstm = TreeLSTM(
            config["H"], len(WORD.vocab) + 100, len(LABEL.vocab)
        ).cuda()
        for p in tree_lstm.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        model = SpanLSTM(NT, len(WORD.vocab), config["H"]).cuda()
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        wandb.watch((model, tree_lstm))
        print(wandb.config)
        tree = run_train(train_iter, valid_iter, model, tree_lstm, V)
    else:
        print("loading")
        model, tree_lstm = torch.load("cp.yoyo.model")
        print(valid_sup(valid_iter, model, tree_lstm, V))


main()
