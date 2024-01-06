from grail.kge.model import KGEModel
from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, kgemodel: KGEModel, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.kgemodel = kgemodel

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def forward(self, data):
        g, rel_labels = data
        g.ndata['h'] = self.gnn(g)

        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        
        # Check if kgemodel is not None
        if self.kgemodel is not None:
            (kge_q_embs, kge_r_embs) = self.kgemodel(torch.cat([head_ids, rel_labels, tail_ids], dim=0), decompose=True)
            kge_embs = torch.cat([kge_q_embs.view(-1, self.params.emb_dim), kge_r_embs.view(-1, self.params.emb_dim)], dim=1)
        else:
            kge_embs = torch.zeros_like(head_embs).view(-1, self.params.num_gcn_layers * self.params.emb_dim)

        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               self.rel_emb(rel_labels), kge_embs], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels), kge_embs], dim=1)

        output = self.fc_layer(g_rep)
        return output
