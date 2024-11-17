# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 23:13:45 2022

@author: Shiyu
"""

# model
import torch.nn as nn
import torch.nn.functional as F
import torch

from dgl import DGLGraph
import networkx as nx
from dgl.nn.pytorch import GATConv


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.MLP_GCN = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ELU()
            ).cuda()
        
    def forward(self, A, X):
        X_tmp = torch.matmul(A, X)
        h = self.MLP_GCN(X_tmp)
        return h


class gemGAT(nn.Module):
    def __init__(self, ngene_in, ngene_out, nhid, nhidatt, nheads):
        super(gemGAT, self).__init__()
        
        self.ngene_in = ngene_in
        self.ngene_out = ngene_out
        self.nhid = nhid
        self.nheads = nheads
        self.nhidatt = nhidatt

        self.pred_in = nn.Sequential(
            nn.Linear(self.nhidatt, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 16),
            nn.ELU(),
            nn.Linear(16, 4),
            nn.ELU(),
            nn.Linear(4, 1)
            ).cuda()

        self.attentions = GATConv(1, self.nhidatt, self.nheads)
        self.attentions2 = GATConv(self.nhidatt, self.nhidatt, self.nheads)
        self.attentions3 = GATConv(self.nhidatt, self.nhidatt, self.nheads)
        self.attentions4 = GATConv(self.nhidatt, self.nhidatt, self.nheads)
        self.out_att = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        self.out_att2 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        self.out_att3 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        self.out_att4 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)

        # Link prediction
        self.attentions_linkpred1 = GATConv(1, self.nhidatt, self.nheads)
        self.attentions_linkpred2 = GATConv(self.nhidatt, self.nhidatt, self.nheads)
        self.out_att_linkpred1 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        self.out_att_linkpred2 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        self.pred_link = nn.Sequential(
            nn.Linear(self.nhidatt, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128)
            ).cuda()
        
        

        # Semi prediction
        self.attentions_semi1 = GATConv(1, self.nhidatt, self.nheads)
        self.attentions_semi2 = GATConv(self.nhidatt, self.nhidatt, self.nheads)
        
        self.out_att_semi1 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)
        self.out_att_semi2 = GATConv(self.nhidatt * self.nheads, self.nhidatt, 1)

        self.pred_out = nn.Sequential(
            nn.Linear(self.nhidatt, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 16),
            nn.ELU(),
            nn.Linear(16, 4),
            nn.ELU(),
            nn.Linear(4, 1)
            ).cuda()
        
    def forward(self, g1, g2, g3, X):
        X1 = X[:self.ngene_out].view(-1, 1).cuda()

        z = X1.cuda()
        
        zpred = self.attentions(g1, z)
        zpred = zpred.view(zpred.shape[0], -1)
        zpred = F.elu(self.out_att(g1, zpred))
        
        zpred = self.attentions2(g1, zpred)
        zpred = zpred.view(zpred.shape[0], -1)
        zpred = F.elu(self.out_att2(g1, zpred))
        
        zpred = self.attentions3(g2, zpred)
        zpred = zpred.view(zpred.shape[0], -1)
        zpred = F.elu(self.out_att3(g2, zpred))
        
        zpred = self.attentions4(g2, zpred)
        zpred = zpred.view(zpred.shape[0], -1)
        zpred = F.elu(self.out_att4(g2, zpred))
        
        # In-network prediction
        g_in_pred = self.pred_in(zpred).view(-1, 1)[:self.ngene_in, :]
        
        # Impute out-network gene expression
        g_all = torch.cat([g_in_pred, X[self.ngene_in:].view(-1, 1).cuda()], dim = 0)
        
        # Semi-supervised link prediction
        zsemi_lp = self.attentions_linkpred1(g3, g_all)
        zsemi_lp = zsemi_lp.view(zsemi_lp.shape[0], -1)
        zsemi_lp = F.elu(self.out_att_linkpred1(g3, zsemi_lp))
        zsemi_lp = self.attentions_linkpred2(g3, zsemi_lp)
        zsemi_lp = zsemi_lp.view(zsemi_lp.shape[0], -1)
        zsemi_lp = self.out_att_linkpred2(g3, zsemi_lp)
        zsemi_lp = zsemi_lp.view(zsemi_lp.shape[0], -1)
        zsemi_lp = self.pred_link(zsemi_lp)

        # The whole adjacency matrix
        zsemi_lp1 = zsemi_lp[:self.ngene_out, :]
        zsemi_lp2 = zsemi_lp[self.ngene_out:, :]
        A_semi_ori = F.sigmoid(torch.matmul(zsemi_lp1, torch.transpose(zsemi_lp1, 0, 1))).view(zsemi_lp1.shape[0], -1)
        A_semi1 = F.sigmoid(torch.matmul(zsemi_lp1, torch.transpose(zsemi_lp2, 0, 1))).view(zsemi_lp1.shape[0], -1)
        A_semi2 = F.sigmoid(torch.matmul(zsemi_lp2, torch.transpose(zsemi_lp2, 0, 1))).view(zsemi_lp2.shape[0], -1)
        
        
        zsemi = self.attentions_semi1(g3, g_all)
        zsemi = zsemi.view(zsemi.shape[0], -1)
        zsemi = F.elu(self.out_att_semi1(g3, zsemi))
        zsemi = self.attentions_semi2(g3, zsemi)
        zsemi = zsemi.view(zsemi.shape[0], -1)
        zsemi = self.out_att_semi2(g3, zsemi)
        
        g_pred_all = self.pred_out(zsemi)

        return g_in_pred, g_pred_all, A_semi1, A_semi2, A_semi_ori
