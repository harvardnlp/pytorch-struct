# -*- coding: utf-8 -*-
# wandb login 7cd7ade39e2d850ec1cf4e914d9a148586a20900
from torch_struct import CKY_CRF, CKY, LogSemiring, MaxSemiring, SampledSemiring, EntropySemiring
import torchtext.data as data 
from torch_struct.data import ListOpsDataset, TokenBucket
from torch_struct.networks import NeuralCFG, TreeLSTMCell
import torch
import dgl
import pickle
import torch.nn as nn
import wandb
from torch_struct import MultiSampledSemiring
import networkx
import time


config = {"method": "reinforce", "baseline": "mean", "opt": "adadelta", 
          "lr_struct": 0.1, "lr_params": 1, "train_model":True, 
          "var_norm": False, "entropy": 0.001, "v": 3, "RL_K": 10 }

wandb.init(project="pytorch-struct", config=config)

def tree_post(v):
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

class ListOpsDataset(data.Dataset):
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
                        nt = last if stack[-1] == cur else "nt"
                        spans.append((stack[-1], cur-1, w))
                        stack = stack[:-1]
                    else:
                        last = w
                        spans.append((cur, cur, w))
                        cur += 1
                examples.append(data.Example.fromlist((words, label, spans), fields))
        super(ListOpsDataset, self).__init__(examples, fields, **kwargs)


def run(graph, cell, iou, h, c, topo=None):
    g = graph
    g.register_message_func(cell.message_func)
    g.register_reduce_func(cell.reduce_func)
    g.register_apply_node_func(cell.apply_node_func)
    # feed embedding
    g.ndata["iou"] = iou
    g.ndata["h"] = h
    g.ndata["c"] = c
    # propagate
    if topo is None:
        dgl.prop_nodes_topo(g)
    else:
        g.prop_nodes(topo)
    return g.ndata.pop("h")


class Res(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.u1 = nn.Linear(H, H)
        self.u2 = nn.Linear(H, H)

        self.v1 = nn.Linear(H, H)
        self.v2 = nn.Linear(H, H)
        self.w = nn.Linear(H, H)

    def forward(self, y):
        y = self.w(y)
        y = y + torch.relu(self.v1(torch.relu(self.u1(y))))
        return y + torch.relu(self.v2(torch.relu(self.u2(y))))
    
class LSTMParse(torch.nn.Module):
    def __init__(self, NT, V, H):
        super().__init__()
        self.H = H
        self.V = V
        self.NT = NT
        self.emb = torch.nn.Embedding(V, H)
        self.lstm = torch.nn.LSTM(H, H, batch_first=True, bidirectional=True)
        self.res = Res(2*H)
        self.proj = torch.nn.Linear(2*H, NT)
    
    def forward(self, words, lengths):
            #packed_words = torch.nn.utils.rnn.pack_padded_sequence(words, lengths, batch_first=True, enforce_sorted=False)
        self.emb.weight[1].data.zero_()
        batch, N = words.shape
        f, ba = self.lstm(self.emb(words))[0].chunk(2, dim=2)
        a = torch.zeros(batch, N, N, H).type_as(f)
        b = torch.zeros(batch, N, N, H).type_as(f)
        a[:, :, :N-1] = f[:, 1:].view(batch, 1, N-1, self.H) - f[:].view(batch, N, 1,   self.H)
        b[:, 1:, :] = ba[:, :N-1].view(batch, N-1,  1,  self.H) - ba.view(batch, 1, N,   self.H)
        out = self.proj(self.res(torch.cat([a, b], dim=-1)))
        return out

#model = LSTMParse(1, len(WORD.vocab), 100).cuda()
#for i, ex in enumerate(valid_iter):
#    words, lengths = ex.word
#    print(words.shape)
#    print(model(words, lengths).shape)
#    break


WORD = data.Field(include_lengths=True, batch_first=True, eos_token=None, init_token=None)
LABEL = data.Field(sequential=False, batch_first=True)
TREE = data.RawField(postprocessing=tree_post(WORD))
TREE.is_target=False
train = ListOpsDataset("data/train_d20s.tsv", (("word", WORD), ("label", LABEL), ("tree", TREE)),
                       filter_pred=lambda x: 5 < len(x.word) < 50)
WORD.build_vocab(train)
LABEL.build_vocab(train)
valid = ListOpsDataset("data/test_d20s.tsv", (("word", WORD), ("label", LABEL), ("tree", TREE)),
                       filter_pred=lambda x: 5 < len(x.word) < 150)

train_iter = TokenBucket(train, 
    batch_size=1500,
    device="cuda:0", key=lambda x: len(x.word))
train_iter.repeat = False
valid_iter = data.BucketIterator(train, 
    batch_size=50,
    device="cuda:0")

H = 100
tree_lstm = TreeLSTMCell(H, H).cuda()
emb = torch.nn.Embedding(len(WORD.vocab) + 100, 100).cuda()
out = torch.nn.Linear(H, len(LABEL.vocab)).cuda()

params = list(emb.parameters()) + list(tree_lstm.parameters()) + list(out.parameters()) 
for p in params:
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

if not config["train_model"]:
    import pickle
    par = pickle.load(open("params.pkl", "rb"))
    for i, p in enumerate(list(emb.parameters()) + list(tree_lstm.parameters()) + list(out.parameters())):
        p.data.copy_(par[i])
    valid_sup(valid_iter)

g_time = time.time()
last = ""
total_time = {}
def t(s):
    global g_time, last, total_time
    if last:
        total_time.setdefault(last, 0)
        total_time[last] += time.time() - g_time
    g_time = time.time()
    last = s

def detach(t):
    return t.detach()
    return tuple((a.detach() for a in t))

NT = 1
T = len(WORD.vocab)
V = T 
model = LSTMParse(NT, len(WORD.vocab), H).cuda()
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
struct = CKY_CRF
joint_params = params
opt_struct = torch.optim.Adadelta(list(model.parameters()), lr=config["lr_struct"])
opt_params = torch.optim.Adadelta(joint_params, lr=config["lr_params"])

wandb.watch(model)


def sample_baseline_a(reward_fn, phi, lengths):
    sample = struct(SampledSemiring).marginals(phi, lengths=lengths)
    argmax = struct(MaxSemiring).marginals(phi, lengths=lengths)        
    sampled_tree = struct().from_parts(detach(sample))[0].cpu()
    argmax_tree = struct().from_parts(detach(argmax))[0].cpu()
    sample_score = reward_fn(sampled_tree)
    max_score = reward_fn(argmax_tree)
    reward = sample_score - max_score
    reward = reward.detach()
    return [sample], torch.stack([reward]), sample_score, max_score


def sample_baseline_b(reward_fn, phi, lengths, K=5):
    t("sample")
    sample = struct(MultiSampledSemiring).marginals(phi, lengths=lengths)
    sample = detach(sample)
    
    t("construct")
    trees = []
    samples = []
    for k in range(K):
        tmp_sample = MultiSampledSemiring.to_discrete(sample, k+1)
        samples.append(tmp_sample)
        sampled_tree = struct().from_parts(tmp_sample)[0].cpu()
        trees.append(sampled_tree)
    structs = torch.stack(samples)
    argmax = struct(MaxSemiring).marginals(phi, lengths=lengths)    
    argmax_tree = struct().from_parts(detach(argmax))[0].cpu()
    trees.append(argmax_tree)

    t("use")
    sample_score = reward_fn(torch.cat(trees), K+1)

    t("finish")
    total = sample_score[:-1].mean(dim=0)
    # for k in range(K):
    #     samples.append([trees[k][1], sample_scores[k]])
    #     if k == 0:
    #         total = sample_score.clone()
    #     else:
    #         total += sample_score
    max_score = sample_score[-1].clone().detach()
    rewards = sample_score[:-1] - max_score.view(1, sample_score.shape[1])
    return structs, rewards, total, max_score

def tree_model(trees, lengths):
    (n_nodes, a, b, label), indices, topo = CKY.to_networkx(trees.cpu())
    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    g.add_edges(a, b)
    # g.from_networkx(graph, node_attrs=['label'])

    t("ftree")
    h = torch.zeros(n_nodes, H, device="cuda:0")
    c = torch.zeros(n_nodes, H, device="cuda:0")
    iou = emb(label.cuda())

    g = run(g, tree_lstm, tree_lstm.W_iou(iou), h, c, topo=topo)
    final = torch.stack([g[indices[i, 0][0]]  for i, l in enumerate(lengths)])
    final = out(final).log_softmax(dim=-1)
    t("ex")
    return final


def valid_show():
    for i, ex in enumerate(valid_iter):
        words, lengths = ex.word
        label = ex.label
        batch = label.shape[0]
        words = words.cuda()
        out = [WORD.vocab.itos[w.item()] for w in words[0]]
        print(" ".join(out))
        def show(tree):
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
                    print("(", end=" ")
                print(w, end=" ")
                for _ in range(end.get(i, 0)):
                    print(")", end=" ")
            print()
        show(ex.tree[0])
        phi = model(words, lengths)
        argmax = struct(MaxSemiring).marginals(phi, lengths=lengths)    
        argmax_tree = struct().from_parts(detach(argmax))[0].cpu()
        show(argmax_tree[0])
        break

def valid_sup(valid_iter):
    total = 0
    correct = 0
    for i, ex in enumerate(valid_iter):
        words, lengths = ex.word
        trees = ex.tree
        label = ex.label
        batch = label.shape[0]
        words = words.cuda()
        _, N = words.shape

        def tree_reward(spans):
            spans[:, torch.arange(N), torch.arange(N)] = 0
            new_spans = torch.zeros(batch, N, N, 1 + V, device=spans.device, dtype=spans.dtype)
            new_spans[:, :, :, :1] = spans
            new_spans[:, torch.arange(N), torch.arange(N), :].fill_(0)
            new_spans[:, torch.arange(N), torch.arange(N), 1:] = torch.nn.functional.one_hot(words, V).float().cuda()
            #torch.nn.functional.one_hot(words, V).float().cuda()
            t("tree")
            _, am = tree_model(new_spans, lengths).max(-1)
            return (label == am).sum(), label.shape[0]
           
        words = words.cuda()
        phi = model(words, lengths)
        argmax = struct(MaxSemiring).marginals(phi, lengths=lengths)    
        argmax_tree = struct().from_parts(detach(argmax))[0]
        score, tota = tree_reward(argmax_tree)
        total += int(tota)
        correct += score
        
        if i == 25: break
    print(correct.item() / float(total), correct, total)
    return correct.item() / float(total)

def train(train_iter):
    model.train()
    losses = []
    alpha = 0.90
    for epoch in range(50):
        print("epoch", epoch)
        running_reward = None
        for i, ex in enumerate(train_iter):
            opt_params.zero_grad()
            words, lengths = ex.word
            label = ex.label
            batch = label.shape[0]
            _, N = words.shape

            def tree_reward(spans, K):
                spans[:, torch.arange(N), torch.arange(N)] = 0
                new_spans = torch.zeros(K, batch, N, N, 1 + V).cuda()
                new_spans[:, :, :, :, :1] = spans.view(K, batch,  N, N, 1)
                new_spans[:, :, torch.arange(N), torch.arange(N), :].fill_(0)
                new_spans[:, :, torch.arange(N), torch.arange(N), 1:] = torch.nn.functional.one_hot(words, V).float().cuda().view(1, batch, N, V)
                
                new_spans = new_spans.view(batch*K, N, N, 1 + V)
                ret = tree_model(new_spans, torch.cat([lengths for _ in range(K)]))
                ret = ret.view(K, batch, -1)                
                return -ret[:, torch.arange(batch), label].view(K, batch)
            
            words = words.cuda()
            phi = model(words, lengths)
            if config["baseline"] == "mean":
                structs, rewards, score, max_score = sample_baseline_b(tree_reward, phi, lengths, K=config["RL_K"])
            if config["baseline"] == "sct":
                structs, rewards, score, max_score = sample_baseline_a(tree_reward, phi, lengths)
            if config["train_model"]:
                t("backward")
                (score.mean()).backward()
                torch.nn.utils.clip_grad_norm_(parameters = joint_params, max_norm = 0.5, norm_type=float("inf"))
                opt_params.step()
                opt_params.zero_grad()
            
            if config["method"] == "reinforce":            
                opt_struct.zero_grad()
                obj = [] 
                t("policy")
                log_partition, entropy = struct(EntropySemiring).sum(phi, lengths=lengths, _raw=True).unbind()
                # assert rewards.shape[0] == len(structs)
                # for sample, reward in zip(structs, rewards):
                    #if running_reward is None:
                    #    running_reward = reward.var().detach()
                    #else:
                    #    running_reward = running_reward * alpha + reward.var() * (1.0 - alpha)
                    #    reward = reward / running_reward.sqrt().clamp(min=1.0)
                rewards = rewards.detach()
                s = structs.shape
                r = struct().score(phi.unsqueeze(0), structs, batch_dims=[0,1]) - log_partition.unsqueeze(0)

                obj = rewards.mul(r).mean(-1).mean(-1)

                policy = obj - config["entropy"] * entropy.mean()
                (policy).backward()
                torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 0.5, norm_type=float("inf"))
                opt_struct.step()
                opt_struct.zero_grad()

            if config["method"] == "ppo":
                # Run PPO
                old = None
                for p in range(10):
                    opt_struct.zero_grad()
                    obj = [] 
                    t("model")
                    phi = model(words, lengths)
                    for sample, reward in zip(structs, rewards):
                        #if running_reward is None:
                        #    running_reward = reward.var().detach()
                        #else:
                        #    if p == 0:
                        #        running_reward = running_reward * alpha + reward.var() * (1.0 - alpha)
                        #    reward = reward / running_reward.sqrt().clamp(min=1.0)
                        t("dp")
                        reward = reward.detach()
                        log_partition, entropy = struct(EntropySemiring).sum(phi, lengths=lengths, _raw=True).unbind()
                        t("add")
                        cur = struct().score(phi, sample.cuda()) - log_partition 
                        
                        if p == 0:
                            old = cur.clone().detach()
                        r = (cur - old).exp()  
                        clamped_r =  torch.clamp(r, 0.98, 1.02)                     
                        obj.append(torch.max(reward.mul(r), reward.mul(clamped_r)).mean()) 
                    t("rest")
                    policy = torch.stack(obj).mean(dim=0) - config["entropy"] * entropy.mean()
                    (policy).backward()
                    t("update")
                    torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 0.5, norm_type=float("inf"))
                    opt_struct.step()
                    opt_struct.zero_grad()
        
            losses.append(-max_score.mean().detach())
            if i % 50 == 9:            
                print(torch.tensor(losses).mean(), words.shape)
                print("Round")
                print("Entropy", entropy.mean())
                print("Reward", rewards.mean())
                print("Running Reward", running_reward)
                if i % 200 == 9:
                    valid_loss = valid_sup(valid_iter)
                else:
                    print(valid_loss)
                valid_show()
                wandb.log({"entropy": entropy.mean(), "valid_loss": valid_loss, 
                           "reward": rewards.mean(),
                           "reward_var": rewards.var(),
                           "loss" : torch.tensor(losses).mean()})

                for k in total_time:
                    print(k, total_time[k])
                losses = []

tree = train(train_iter)


# opt = torch.optim.Adam(params, lr=0.001, betas=[0.75, 0.999])

# def tree_model(trees, lengths):
#     graph, indices = CKY.to_networkx(trees.cpu())
#     g = dgl.DGLGraph()
#     g.from_networkx(graph, node_attrs=['label'])

#     h = torch.zeros(len(graph.nodes), H, device="cuda:0")
#     c = torch.zeros(len(graph.nodes), H, device="cuda:0")
#     iou = emb(g.ndata["label"].cuda())
#     g = run(g, tree_lstm, tree_lstm.W_iou(iou), h, c)
#     final = torch.stack([g[indices[i, 0, l.item()-1]]  for i, l in enumerate(lengths)])
#     final = out(final).log_softmax(dim=-1)
#     return final

# def valid_sup(valid_iter):
#     total = 0
#     correct = 0
#     for i, ex in enumerate(valid_iter):
#         words, lengths = ex.word
#         trees = ex.tree
#         label = ex.label
#         batch = label.shape[0]
#         words = words.cuda()
#         final = tree_model(trees, lengths)
#         _, argmax = final.max(-1)
#         total += batch
#         correct += (argmax == label).sum().item()
#         if i == 25: break
#     print(correct / float(total), correct, total)
# def train_sup(train_iter):
#     tree_lstm.train()
#     losses = []
#     for epoch in range(10):
#         for i, ex in enumerate(train_iter):
#             opt.zero_grad()
#             words, lengths = ex.word
#             trees = ex.tree
#             label = ex.label
#             batch = label.shape[0]
#             words = words.cuda()
#             final = tree_model(trees, lengths)
#             loss = final[torch.arange(batch), label].mean()
#             (-loss).backward()
#             torch.nn.utils.clip_grad_norm_(params, 5.0)
#             opt.step()
#             losses.append(loss.detach())
#             if i % 500 == 1:            
#                 pickle.dump(params, open("params.pkl", "wb"))
#                 print(-torch.tensor(losses).mean(), words.shape)
#                 valid_sup(valid_iter)
#                 losses = []

#train_sup(train_iter)
#exit()

