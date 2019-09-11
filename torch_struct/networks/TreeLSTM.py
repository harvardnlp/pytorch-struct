# From DGL documentation

import torch as th
import torch.nn as nn
import dgl


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox["h"].size())
        c = th.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_cat), "c": c}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * th.tanh(c)
        return {"h": h, "c": c}


def run(cell, graph, iou, h, c):
    g = graph
    g.register_message_func(cell.message_func)
    g.register_reduce_func(cell.reduce_func)
    g.register_apply_node_func(cell.apply_node_func)
    # feed embedding
    g.ndata["iou"] = iou
    g.ndata["h"] = h
    g.ndata["c"] = c
    # propagate
    dgl.prop_nodes_topo(g)
    return g.ndata.pop("h")
