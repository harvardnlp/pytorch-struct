# -*- coding: utf-8 -*-
# wandb login 7cd7ade39e2d850ec1cf4e914d9a148586a20900
from torch_struct import CKY_CRF, CKY, LogSemiring, MaxSemiring, SampledSemiring, EntropySemiring, SelfCritical
import torchtext.data as data 
from torch_struct.data import ListOpsDataset, TokenBucket
from torch_struct.networks import NeuralCFG, TreeLSTM, SpanLSTM
import torch
import torch.nn as nn
import wandb
from torch_struct import MultiSampledSemiring

config = {"method": "reinforce", "baseline": "mean", "opt": "adadelta", 
          "lr_struct": 0.1, "lr_params": 1, "train_model":True, 
          "var_norm": False, "entropy": 0.001, "v": 3, "RL_K": 5, 
          "H" : 100,
}

wandb.init(project="pytorch-struct", config=config)

def clip(p):
    torch.nn.utils.clip_grad_norm_(parameters = p, 
                                   max_norm = 0.5, norm_type=float("inf"))


def expand_spans(spans, words, K, V):
    batch, N = words.shape
    spans[:, torch.arange(N), torch.arange(N)] = 0
    new_spans = torch.zeros(K, batch, N, N, 1 + V).cuda()
    new_spans[:, :, :, :, :1] = spans.view(K, batch,  N, N, 1)
    new_spans[:, :, torch.arange(N), torch.arange(N), :].fill_(0)
    new_spans[:, :, torch.arange(N), torch.arange(N), 1:] = \
        torch.nn.functional.one_hot(words, V).float().cuda().view(1, batch, N, V)
    new_spans = new_spans.view(batch*K, N, N, 1 + V)
    return new_spans


def valid_sup(valid_iter, model, tree_lstm):
    total = 0
    correct = 0
    struct = CKY_CRF
    for i, ex in enumerate(valid_iter):
        words, lengths = ex.word
        trees = ex.tree
        label = ex.label
        words = words.cuda()

        def tree_reward(spans):
            new_spans = expand_spans(spans.unsqueeze(0))
            g, labels, indices, topo = TreeLSTM.spans_to_dgl(new_spans)
            _, am = tree_lstm(g, labels, indices, topo, lengths).max(-1)
            return (label == am).sum(), label.shape[0]
           
        words = words.cuda()
        phi = model(words, lengths)
        argmax = struct(MaxSemiring).marginals(phi, lengths=lengths)    
        argmax_tree = struct().from_parts(argmax.detach())[0]
        score, tota = tree_reward(argmax_tree)
        total += int(tota)
        correct += score
        
        if i == 25: break
    print(correct.item() / float(total), correct, total)
    return correct.item() / float(total)


def run_train(train_iter, valid_iter, model, tree_lstm, V):
    opt_struct = torch.optim.Adadelta(list(model.parameters()), lr=config["lr_struct"])
    opt_params = torch.optim.Adadelta(list(tree_lstm.parameters()), lr=config["lr_params"])

    model.train()
    losses = []
    struct = CKY_CRF

    for epoch in range(50):
        print("Epoch", epoch)

        for i, ex in enumerate(train_iter):
            words, lengths = ex.word
            label = ex.label
            batch = label.shape[0]
            _, N = words.shape
            words = words.cuda()


            def tree_reward(spans, K):
                new_spans = expand_spans(spans, words, K, V)
                g, labels, indices, topo = TreeLSTM.spans_to_dgl(new_spans)
                ret = tree_lstm(g, labels, indices, topo, torch.cat([lengths for _ in range(K)]))
                ret = ret.view(K, batch, -1)                
                return -ret[:, torch.arange(batch), label].view(K, batch)
            
            sc = SelfCritical(CKY_CRF, tree_reward)
            phi = model(words, lengths)
            structs, rewards, score, max_score = sc.forward(phi, lengths, K=config["RL_K"], )
            
            if config["train_model"]:
                opt_params.zero_grad()
                score.mean().backward()
                clip(tree_lstm.parameters())
                opt_params.step()
            
            if config["method"] == "reinforce":            
                opt_struct.zero_grad()
                log_partition, entropy = struct(EntropySemiring).sum(phi, lengths=lengths, _raw=True).unbind()
                r = struct().score(phi.unsqueeze(0), structs, batch_dims=[0,1]) \
                     - log_partition.unsqueeze(0)
                obj = rewards.mul(r).mean(-1).mean(-1)
                policy = obj - config["entropy"] * entropy.mean()
                policy.backward()
                clip(model.parameters())
                opt_struct.step()     
            losses.append(-max_score.mean().detach())



            # DEBUG
            if i % 50 == 9:            
                print(torch.tensor(losses).mean(), words.shape)
                print("Round")
                print("Entropy", entropy.mean())
                print("Reward", rewards.mean())
                if i % 200 == 9:
                    valid_loss = valid_sup(valid_iter, model, tree_lstm)
                else:
                    print(valid_loss)
                # valid_show()
                wandb.log({"entropy": entropy.mean(), "valid_loss": valid_loss, 
                           "reward": rewards.mean(),
                           "reward_var": rewards.var(),
                           "loss" : torch.tensor(losses).mean()})
                losses = []


def main():
    WORD = data.Field(include_lengths=True, batch_first=True, eos_token=None, init_token=None)
    LABEL = data.Field(sequential=False, batch_first=True)
    TREE = data.RawField(postprocessing=ListOpsDataset.tree_field(WORD))
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

    NT = 1
    T = len(WORD.vocab)
    V = T 

    tree_lstm = TreeLSTM(config["H"],
                         len(WORD.vocab) + 100, len(LABEL.vocab)).cuda()
    for p in tree_lstm.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    model = SpanLSTM(NT, len(WORD.vocab), config["H"]).cuda()
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    struct = CKY_CRF

    wandb.watch(model)
    tree = run_train(train_iter, valid_iter, model, tree_lstm, V)

main()



# def valid_show():
#     for i, ex in enumerate(valid_iter):
#         words, lengths = ex.word
#         label = ex.label
#         batch = label.shape[0]
#         words = words.cuda()
#         out = [WORD.vocab.itos[w.item()] for w in words[0]]
#         print(" ".join(out))
#         def show(tree):
#             start = {}
#             end = {}    
#             for i, j, _ in tree.nonzero():
#                 i = i.item()
#                 j = j.item()
#                 start.setdefault(i, -1)
#                 end.setdefault(j, -1)
#                 start[i] += 1
#                 end[j] += 1
#             for i, w in enumerate(out):
#                 for _ in range(start.get(i, 0)):
#                     print("(", end=" ")
#                 print(w, end=" ")
#                 for _ in range(end.get(i, 0)):
#                     print(")", end=" ")
#             print()
#         show(ex.tree[0])
#         phi = model(words, lengths)
#         argmax = struct(MaxSemiring).marginals(phi, lengths=lengths)    
#         argmax_tree = struct().from_parts(detach(argmax))[0].cpu()
#         show(argmax_tree[0])
#         break


            # if config["method"] == "ppo":
            #     # Run PPO
            #     old = None
            #     for p in range(10):
            #         opt_struct.zero_grad()
            #         obj = [] 
            #         t("model")
            #         phi = model(words, lengths)
            #         for sample, reward in zip(structs, rewards):
            #             #if running_reward is None:
            #             #    running_reward = reward.var().detach()
            #             #else:
            #             #    if p == 0:
            #             #        running_reward = running_reward * alpha + reward.var() * (1.0 - alpha)
            #             #    reward = reward / running_reward.sqrt().clamp(min=1.0)
            #             t("dp")
            #             reward = reward.detach()
            #             log_partition, entropy = struct(EntropySemiring).sum(phi, lengths=lengths, _raw=True).unbind()
            #             t("add")
            #             cur = struct().score(phi, sample.cuda()) - log_partition 
                        
            #             if p == 0:
            #                 old = cur.clone().detach()
            #             r = (cur - old).exp()  
            #             clamped_r =  torch.clamp(r, 0.98, 1.02)                     
            #             obj.append(torch.max(reward.mul(r), reward.mul(clamped_r)).mean()) 
            #         t("rest")
            #         policy = torch.stack(obj).mean(dim=0) - config["entropy"] * entropy.mean()
            #         (policy).backward()
            #         t("update")
            #         torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 0.5, norm_type=float("inf"))
            #         opt_struct.step()
            #         opt_struct.zero_grad()

# def sample_baseline_b(reward_fn, phi, lengths, K=5):
#     t("sample")
#     sample = struct(MultiSampledSemiring).marginals(phi, lengths=lengths)
#     sample = detach(sample)
    
#     t("construct")
#     trees = []
#     samples = []
#     for k in range(K):
#         tmp_sample = MultiSampledSemiring.to_discrete(sample, k+1)
#         samples.append(tmp_sample)
#         sampled_tree = struct().from_parts(tmp_sample)[0].cpu()
#         trees.append(sampled_tree)
#     structs = torch.stack(samples)
#     argmax = struct(MaxSemiring).marginals(phi, lengths=lengths)    
#     argmax_tree = struct().from_parts(detach(argmax))[0].cpu()
#     trees.append(argmax_tree)

#     t("use")
#     sample_score = reward_fn(torch.cat(trees), K+1)

#     t("finish")
#     total = sample_score[:-1].mean(dim=0)
#     # for k in range(K):
#     #     samples.append([trees[k][1], sample_scores[k]])
#     #     if k == 0:
#     #         total = sample_score.clone()
#     #     else:
#     #         total += sample_score
#     max_score = sample_score[-1].clone().detach()
#     rewards = sample_score[:-1] - max_score.view(1, sample_score.shape[1])
#     return structs, rewards, total, max_score

