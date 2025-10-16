import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge,GCN2,GCN3
from functools import partial

from ogb.linkproppred import Evaluator
from ogbdataset import loaddataset
from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
import time
import torch
from torch_sparse import SparseTensor

import torch_sparse

from pygho import SparseTensor as pSparseTensor
from pygho.backend.Spspmm import spsphadamard, spspmm
from pygho.backend.Spmm import spmm

from torch.amp.grad_scaler import GradScaler
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T

from functools import partial

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter

import time
from typing import Iterable
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T

from functools import partial

from ogb.linkproppred import Evaluator

from torch.utils.tensorboard import SummaryWriter

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def get_cn1_cn2(adj,tedge):
    Ei = adj.index_select([0], tedge[0].unsqueeze(0))
            
    Ej = adj.index_select([0], tedge[1].unsqueeze(0))
    cn1 = spsphadamard(Ei, Ej)
    Ej2 = spspmm(Ej, 1, adj, 0)
    del Ej
    cn2 = spsphadamard(Ei, Ej2)
    del Ei, Ej2

    cn1=cn1.to_torch_sparse_coo()
    cn2=cn2.to_torch_sparse_coo()

    num_nodes = cn1.shape[1] 
    num_edges=cn1.shape[0]

    row1, col1 = cn1.indices()
    row2, col2 = cn2.indices()

    value1 = cn1.values()
    value2 = cn2.values()


    cn1 = torch_sparse.SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=(num_edges, num_nodes))
    cn2 = torch_sparse.SparseTensor(row=row2, col=col2, value=value2, sparse_sizes=(num_edges, num_nodes))

    return cn1,cn2

def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float] = [],
          linkbatchsize: int = None,
          args=None):



    if linkbatchsize is None:
        linkbatchsize = (batch_size + 31) // 32

    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = []

    adjmask = torch.ones_like(source_edge, dtype=torch.bool)

    for perm in PermIterator(source_edge.device, source_edge.shape[0], batch_size):
        optimizer.zero_grad()

        if maskinput:
            adjmask[perm] = 0
            tei = torch.stack((source_edge[adjmask], target_edge[adjmask]), dim=0)
            adj = SparseTensor.from_edge_index(
                tei,
                sparse_sizes=(data.num_nodes, data.num_nodes)
            ).to_device(source_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t

        row, col, val = adj.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        loss = 0
        adj = pSparseTensor(torch.stack((row, col)), val, adj.sizes(), is_coalesced=True)

        
        h0 = model(data.x, adj)
        h = h0.detach().requires_grad_(True)
        torch.cuda.empty_cache()

        source_edge_perm = source_edge[perm]
        target_edge_perm = target_edge[perm]
        source_totallen = source_edge_perm.shape[0]

        for i in range(0, source_totallen, linkbatchsize):
            start, end = i, min(i + linkbatchsize, source_totallen)
            if start == end:  
                continue

            source_tedge = source_edge_perm[start:end]
            target_tedge = target_edge_perm[start:end]
            cn1,cn2=get_cn1_cn2(adj,torch.stack((source_tedge, target_tedge)))
            pos_outs = predictor.multidomainforward(h, adj,cn1,cn2,
                                                    torch.stack((source_tedge, target_tedge)),
                                                    args,
                                                    cndropprobs=cnprobs)

            pos_losss = -(1 / source_totallen) * F.logsigmoid(pos_outs).sum()
            pos_losss.backward()  

            loss += pos_losss.item()
            del pos_outs, pos_losss

        neg_edge = torch.randint(0, data.num_nodes, source_edge_perm.size(),
                                 dtype=torch.long, device=h.device)

        source_totallen = neg_edge.shape[0]
        for i in range(0, source_totallen, linkbatchsize):
            start, end = i, min(i + linkbatchsize, source_totallen)
            if start == end:  
                continue

            source_tedge = source_edge_perm[start:end]
            neg_tedge = neg_edge[start:end]
            if source_tedge.size(0) != neg_tedge.size(0):  
                continue

            cn1,cn2=get_cn1_cn2(adj,torch.stack((source_tedge, neg_tedge)))
            neg_outs = predictor.multidomainforward(h, adj,cn1,cn2,
                                                    torch.stack((source_tedge, neg_tedge)),
                                                    args,
                                                    cndropprobs=cnprobs)
            neg_losss = -(1 / source_totallen) * F.logsigmoid(-neg_outs).sum()
            neg_losss.backward()  

            loss += neg_losss.item()
            del neg_outs, neg_losss

       

        optimizer.step()
        del h, h0
        
        total_loss.append(loss)

    total_loss = np.average([_ for _ in total_loss])
    return total_loss






@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input,args):
    model.eval()
    predictor.eval()
    adj = data.full_adj_t
    row, col, val = adj.coo()
    if val is None:
        val = torch.ones_like(row, dtype=torch.float)
        
    adj = pSparseTensor(torch.stack((row, col)), val, adj.sizes(), is_coalesced=True)
    
    h = model(data.x, adj)

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in PermIterator(source.device, source.shape[0], batch_size, False):
            src, dst = source[perm], target[perm]
            cn1,cn2=get_cn1_cn2(adj,torch.stack((src, dst)))
            pos_preds += [predictor(h, adj,cn1,cn2, torch.stack((src, dst)),args).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in PermIterator(source.device, source.shape[0], batch_size, False):
            src, dst_neg = source[perm], target_neg[perm]
            cn1,cn2=get_cn1_cn2(adj,torch.stack((src, dst_neg)))
            neg_preds += [predictor(h, adj,cn1,cn2, torch.stack((src, dst_neg)),args).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
                'y_pred_pos': pos_pred,
                'y_pred_neg': neg_pred,
            })['mrr_list'].mean().item()

    train_mrr = 0.0 
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr, h.cpu()


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--dataset', type=str, default="collab")
    
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--linkbatchsize', type=int, default=8192, help="link-predictor's batch size")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")

    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.3, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.3, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0003, help="learning rate of predictor")
    
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")
    
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")

    
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")

    
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    
    parser.add_argument('--cndeg', type=int, default=-1)
    
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps")
    
    parser.add_argument('--model', choices=convdict.keys())

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")

    parser.add_argument('--cnprob', type=float, default=0)

    parser.add_argument("--adj2byblock", action="store_true", help="whether to get adj^2 by block-multiply")

    parser.add_argument('--sum', type=float, default=0)
    parser.add_argument("--polyfirst", type=int, default=0)
    parser.add_argument("--polysecond", type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    #args = Args()
    print(args, flush=True)
    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    data, split_edge = loaddataset(args.dataset, False, args.load)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "cn8", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor in ["incn1cn1", "sincn1cn1"]:
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []

    for run in range(args.runs):
        set_seed(run)
        bestscore = [0, 0, 0]
        model = GCN3(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp).to(device)

        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}])


        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, args.maskinput, [], args.linkbatchsize,args)

            print(f"trn time {time.time()-t1:.2f} s")
            if True:
                t1 = time.time()
                results= test(model, predictor, data, split_edge, evaluator,
                               args.testbs, args.use_valedges_as_input,args)
                results, h = results[:-1], results[-1]
                print(f"test time {time.time()-t1:.2f} s")
                writer.add_scalars(f"mrr_{run}", {
                        "trn": results[0],
                        "val": results[1],
                        "tst": results[2]
                    }, epoch)

                if True:
                    train_mrr, valid_mrr, test_mrr = results
                    train_mrr, valid_mrr, test_mrr = results
                    if valid_mrr > bestscore[1]:
                        bestscore = list(results) 
                        bestscore = list(results) 
                        if args.save_gemb:
                            torch.save(h, f"gemb/citation2_{args.model}_{args.predictor}.pt")

                    print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_mrr:.2f}%, '
                              f'Valid: {100 * valid_mrr:.2f}%, '
                              f'Test: {100 * test_mrr:.2f}%')
                    print('---', flush=True)
        print(f"best {bestscore}")
        if args.dataset == "citation2":
            ret.append(bestscore)
        else:
            raise NotImplementedError
    ret = np.array(ret)
    print(ret)
    print(f"Final result: {np.average(ret[:, 1])} {np.std(ret[:, 1])} {np.average(ret[:, 2])} {np.std(ret[:, 2])}")

if __name__ == "__main__":
    main()
