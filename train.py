# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:40:14 2022

@author: Shiyu
"""

# import pickle
from model import *
import random
from torch import nn, optim
import numpy as np
import pandas as pd
import dgl

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="Brain-Amygdalaadni")
parser.add_argument('--train', type=str, default="True")
parser.add_argument('--eval', type=str, default="False")
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--nhid', type=int, default=512)
parser.add_argument('--nhidatt', type=int, default=1024)
parser.add_argument('--nheads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)

args = parser.parse_args()
print(f'Agrs: {args}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import dataset
# adj matrix of whole blood
adj = pd.read_csv(f"./data/graph_in_{args.data}.csv", header=0)
adj = adj.iloc[:, 1:]

# adj matrix of another tissue
adj_ms = pd.read_csv(f"./data/graph_out_{args.data}.csv", header=0)
adj_ms = adj_ms.iloc[:, 1:]

# gene expression of whole blood
node_feat = pd.read_csv(f"./data/expr_in_{args.data}.csv", header=0)
# gene expression of another tissue
gene_pred = pd.read_csv(f"./data/expr_out_{args.data}.csv", header=0)
print("Successfully loaded data!")

# data processing
node_feat = torch.from_numpy(node_feat.iloc[:, 1:].to_numpy())
gene_pred = torch.from_numpy(gene_pred.iloc[:, 1:].to_numpy())

adj = torch.tensor(adj.values).float()
adj_ms = torch.tensor(adj_ms.values).float()

adj = adj + torch.eye(adj.shape[0])
adj_ms = adj_ms + torch.eye(adj_ms.shape[0])
adj_ng = adj.shape[0]
adj_ms_ng = adj_ms.shape[0]

adj = adj.numpy()
adj_ms = adj_ms.numpy()

adj_src, adj_dst = np.nonzero(adj)
adj_src = np.append(adj_src, np.arange(adj_ng, adj_ms_ng))
adj_dst = np.append(adj_dst, np.arange(adj_ng, adj_ms_ng))
adj = dgl.graph((adj_src, adj_dst)).to(device)

adj_ms_src, adj_ms_dst = np.nonzero(adj_ms)
adj_semi = adj_ms
adj_ms = dgl.graph((adj_ms_src, adj_ms_dst)).to(device)

adj_all_src = np.append(adj_ms_src, np.arange(adj_ms_ng, gene_pred.shape[0]))
adj_all_dst = np.append(adj_ms_dst, np.arange(adj_ms_ng, gene_pred.shape[0]))
adj_all = dgl.graph((adj_all_src, adj_all_dst)).to(device)

idx = [i for i in range(gene_pred.shape[1])]

# splint training and testing sets
nid = int(len(idx) * 0.9)
idx_train = idx[:nid]
idx_test = idx[nid:]

idx_train = torch.as_tensor(idx_train)
idx_test = torch.as_tensor(idx_test)

node_feat_train = node_feat[:, idx_train].float()
node_feat_test = node_feat[:, idx_test].float()

gene_pred_train = gene_pred[:, idx_train].float()
gene_pred_test = gene_pred[:, idx_test].float()


node_feat_train = torch.log(node_feat_train + 1)
node_feat_test = torch.log(node_feat_test + 1)

gene_pred_train = torch.log(gene_pred_train + 1)
gene_pred_test = torch.log(gene_pred_test + 1)

pos_weight = float(adj_semi.shape[0] * adj_semi.shape[0] - adj_semi.sum()) / adj_semi.sum()
norm = adj_semi.shape[0] * adj_semi.shape[0] / float((adj_semi.shape[0] * adj_semi.shape[0] - adj_semi.sum()) * 2)
weight_mask = torch.from_numpy(adj_semi) == 1
weight_tensor = torch.ones(weight_mask.size(0), weight_mask.size(1)) 
weight_tensor[weight_mask] = pos_weight

if args.train:
    # Training
    print("Training started!")
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # initialize model
    model = gemGAT(ngene_in=adj_ng, ngene_out=adj_ms_ng, nhid = args.nhid, nhidatt=args.nhidatt, nheads=args.nheads).cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3, 5, 10, 15, 20, 30, 40, 50, 80, 100, 300, 500, 700, 800,
                    1000],
        gamma=0.1)
    mseloss = nn.MSELoss()
    
    epochs = args.epoch
    
    loss_record = []
    loss_gene_record = []
    loss_graph_record = []
    loss_l1_record = []
    loss_no_mask = []
    zg_rec = []
    z_rec = []
    
    m = 0
    l1_m = None
    for epoch in range(epochs):
        lr_scheduler.step()
    
        for i in range(len(idx_train)):
            m += 1
    
            y = gene_pred_train[:, i].float()
            X = node_feat_train[:, i].float()
    
            y_pred, y_pred_all, A_semi1, A_semi2, A_semi_ori = model(g1=adj, g2=adj_ms, g3=adj_all, X=X.float().cuda())
    
            y_pred = y_pred.view(-1, 1)
            y_pred_all = y_pred_all.view(-1, 1)
            
            g_pred = torch.cat([y_pred, y_pred_all[adj_ng:]])
            
            loss_graph = norm * F.binary_cross_entropy(A_semi_ori.cuda(), torch.from_numpy(adj_semi).cuda(), weight = weight_tensor.cuda())
            
            loss_gene = mseloss(g_pred, y.view(-1, 1).cuda())
            
            loss = loss_graph + 100*loss_gene
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            print("Epoch: ", epoch, "Iteration: ", i, "m: ", m, "Total loss: ", loss,
                  "Gene prediction loss: ", loss_gene, "Link prediction loss: ", loss_graph)
    
            loss_record.append(loss.detach().cpu().numpy())
            loss_graph_record.append(loss_graph.detach().cpu().numpy())
    
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, f"{args.data}.pt")
        if (epoch + 1) % 20 == 0:
            # Impute graph
            A_semi1[A_semi1 <= 0.99998] = 0
            A_semi1[A_semi1 > 0.99998] = 1
            A_semi2[A_semi2 <= 0.99998] = 0
            A_semi2[A_semi2 > 0.99998] = 1
    
            A_r1, A_c1 = np.nonzero(A_semi1.detach().cpu().numpy())
            A_c1 += adj_ms_ng
    
            A_r2, A_c2 = np.nonzero(A_semi2.detach().cpu().numpy())
            A_r2 += adj_ms_ng
            A_c2 += adj_ms_ng
    
            adj_all.add_edges(A_r2, A_c2)
    
        print("====================================================")
else:
    model = gemGAT(ngene_in=adj_ng, ngene_out=adj_ms_ng, nhid = args.nhid, nhidatt=args.nhidatt, nheads=args.nheads).cuda()

    model.load_state_dict(torch.load(f"{args.data}.pt")['model_state_dict'])
    model.eval()
    print(f"Trained model loaded!")
    
    y_pred_all_rcd = []
    y_all_rcd = []
    with torch.no_grad():
        for i in range(len(idx_test)):
            y = gene_pred_test[:, i].float()
            X = node_feat_test[:, i].float()
            
            y_pred, y_pred_all, A_semi1, A_semi2, A_semi_ori = model(g1=adj, g2=adj_ms, g3=adj_all, X=X.float().cuda())
            y_pred = y_pred.view(-1, 1)
            y_pred_all = y_pred_all.view(-1, 1)
    
            g_pred = torch.cat([y_pred, y_pred_all[adj_ng:]])

            y_pred_all_rcd.append(g_pred)
    print("Inference finished!")
    print("Saving resuls!")
    
    y_all_rcd_df = torch.stack(y_pred_all_rcd, dim = 0)
    y_all_rcd_df = torch.squeeze(y_all_rcd_df)
    y_all_rcd_df = pd.DataFrame(y_all_rcd_df.detach().cpu().numpy())
    y_all_rcd_df.to_csv(f"{args.data}_inference.csv")
    print(f"Saved results to {args.data}_inference.csv")