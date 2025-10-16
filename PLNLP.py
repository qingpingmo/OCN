import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
import time
from ogbdataset import loaddataset
from typing import Iterable
from utils import adjoverlap
from utils import sparse_tensor_multiply


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def auc_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    
    return torch.square(1 - (pos_out - neg_out)).sum()


def hinge_auc_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()


def weighted_auc_loss(pos_out, neg_out, num_neg, weight):
    weight = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (weight*torch.square(1 - (pos_out - neg_out))).sum()


def adaptive_auc_loss(pos_out, neg_out, num_neg, margin):
    margin = torch.reshape(margin, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (torch.square(margin - (pos_out - neg_out))).sum()


def weighted_hinge_auc_loss(pos_out, neg_out, num_neg, weight):
    weight = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (weight*torch.square(torch.clamp(weight - (pos_out - neg_out), min=0))).sum()


def adaptive_hinge_auc_loss(pos_out, neg_out, num_neg, weight):
    weight = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (torch.square(torch.clamp(weight - (pos_out - neg_out), min=0))).sum()


def log_rank_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()


def ce_loss(pos_out, neg_out):
    pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
    return pos_loss + neg_loss


def info_nce_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()



def train(model,
            predictor,
            data,
            split_edge,
            optimizer,
            batch_size,
            maskinput: bool = True,
            cnprobs: Iterable[float]=[],
            alpha: float=None,
            args=None
            ):
        
        
        if alpha is not None:
            predictor.setalpha(alpha)
            
        model.train()
        predictor.train()

        pos_train_edge = split_edge['train']['edge'].to(data.x.device)
        pos_train_edge = pos_train_edge.t()

        total_loss = []
        adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
            
        negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
        for perm in PermIterator(
                    adjmask.device, adjmask.shape[0], batch_size
            ):
            optimizer.zero_grad()
            if maskinput:
                adjmask[perm] = 0
                tei = pos_train_edge[:, adjmask]
                adj = SparseTensor.from_edge_index(tei,
                                    sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                        pos_train_edge.device, non_blocking=True)
                adjmask[perm] = 1
                adj = adj.to_symmetric()
            else:
                adj = data.adj_t
            h = model(data.x, adj)
            edge = pos_train_edge[:, perm]

            spadj = adj.to_torch_sparse_coo_tensor()
            if args.adj2byblock:
                spadj = SparseTensor.from_torch_sparse_coo_tensor(spadj)
                adj2 = sparse_tensor_multiply(spadj, block_size = 1024)
            
            else:
                adj2 = SparseTensor.from_torch_sparse_coo_tensor(spadj @ spadj, False)

            pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    adjoverlap(adj, adj, edge,  False),
                                                    adjoverlap(adj, adj2, edge,  False),
                                                    edge,
                                                    args,
                                                    cndropprobs=cnprobs)

            pos_losss = -F.logsigmoid(pos_outs).mean()
            edge = negedge[:, perm]
            neg_outs = predictor.multidomainforward(h, adj, adjoverlap(adj, adj, edge, cnprobs),adjoverlap(adj, adj2, edge, cnprobs),edge, args,cndropprobs=cnprobs)
            neg_losss = -F.logsigmoid(-neg_outs).mean()

                 
            if args.losstrick == "auc_loss":
                num_neg = 1
                loss = auc_loss(pos_outs, neg_outs, num_neg)
            elif args.losstrick == "hinge_auc_loss": 
                num_neg = 1
                loss = hinge_auc_loss(pos_outs, neg_outs, num_neg)
            elif args.losstrick == "weighted_auc_loss": 
                num_neg = 1
                   
                weight = torch.full_like(pos_outs, args.lossweight).to(pos_outs.device)
                loss = weighted_auc_loss(pos_outs, neg_outs, num_neg, weight=weight)
            elif args.losstrick == "adaptive_auc_loss":
                num_neg = 1
                    
                margin = torch.full_like(pos_outs, args.lossmargin).to(pos_outs.device)
                loss = adaptive_auc_loss(pos_outs, neg_outs, num_neg, margin)
            elif args.losstrick == "weighted_hinge_auc_loss":
                num_neg = 1
                    
                weight = torch.full_like(pos_outs, args.lossweight).to(pos_outs.device)
                loss = weighted_hinge_auc_loss(pos_outs, neg_outs, num_neg, weight=weight)
            elif args.losstrick == "adaptive_hinge_auc_loss": 
                num_neg = 1
                weight = torch.full_like(pos_outs, args.lossweight).to(pos_outs.device)
                loss = adaptive_hinge_auc_loss(pos_outs, neg_outs, num_neg,weight)
            elif args.losstrick == "log_rank_loss": 
                num_neg = 1
                loss = log_rank_loss(pos_outs, neg_outs, num_neg)
            elif args.losstrick == "ce_loss":
                loss = ce_loss(pos_outs, neg_outs)
            elif args.losstrick == "info_nce_loss":
                num_neg = 1
                loss = info_nce_loss(pos_outs, neg_outs, num_neg)
            elif args.losstrick == "simple":
                loss = neg_losss + pos_losss
            else:
                raise ValueError(f"Unknown loss type: {args.losstrick}")

            loss.backward()
            optimizer.step()

            total_loss.append(loss)

            total_loss = np.average([_.item() for _ in total_loss])
            return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input,args):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    spadj = adj.to_torch_sparse_coo_tensor()
    if args.adj2byblock:
        spadj = SparseTensor.from_torch_sparse_coo_tensor(spadj)
        block_size = 1024
        adj2 = sparse_tensor_multiply(spadj, block_size)
        
    else:
        adj2 = SparseTensor.from_torch_sparse_coo_tensor(spadj @ spadj, False)

    pos_train_pred = torch.cat([
        predictor(h, adj,adjoverlap(adj, adj, pos_train_edge[perm].t()),adjoverlap(adj, adj2, pos_train_edge[perm].t()), pos_train_edge[perm].t(), args).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)


    pos_valid_pred = torch.cat([
        predictor(h, adj,adjoverlap(adj, adj, pos_valid_edge[perm].t()),adjoverlap(adj, adj2, pos_valid_edge[perm].t()), pos_valid_edge[perm].t(), args).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    
    neg_valid_pred = torch.cat([
        predictor(h, adj, adjoverlap(adj, adj, neg_valid_edge[perm].t()),adjoverlap(adj, adj2, neg_valid_edge[perm].t()),neg_valid_edge[perm].t(), args).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    
    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, adj,adjoverlap(adj, adj, pos_test_edge[perm].t()),adjoverlap(adj, adj2, pos_test_edge[perm].t()), pos_test_edge[perm].t(), args).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj,adjoverlap(adj, adj, neg_test_edge[perm].t()),adjoverlap(adj, adj2,neg_test_edge[perm].t()), neg_test_edge[perm].t(), args).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']

        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    return results, h.cpu()


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--dataset', type=str, default="collab")
    
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
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
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps ")
   
    parser.add_argument('--model', choices=convdict.keys())

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")

    parser.add_argument('--cnprob', type=float, default=0)

    parser.add_argument("--adj2byblock", action="store_true", help="whether to get adj^2 by block-multiply")
    parser.add_argument('--sum', type=float, default=1)
    parser.add_argument('--losstrick', type=str, default="auc_loss")
    parser.add_argument('--lossmargin', type=float, default=1)
    parser.add_argument('--lossweight', type=float, default=1)
    args = parser.parse_args()
    return args






def main():
   
    print(args, flush=True)

    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
        evaluator = Evaluator(name=f'ogbl-ppa')
    else:
        evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    device = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load)
    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "cn5", "cn7","cn8", "cn9", "cn6", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    
    ret = []

    for run in range(0, args.runs):
        set_seed(run)
        if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load) 
            data = data.to(device)
        bestscore = None
        
       
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        if args.loadx:
            with torch.no_grad():
                model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
            model.xemb[0].weight.requires_grad_(False)
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        if args.loadmod:
            keys = model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
        

        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}])
        
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha,args=args)
            print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if True:
                t1 = time.time()
                results, h = test(model, predictor, data, split_edge, evaluator,
                               args.testbs, args.use_valedges_as_input,args=args)
                print(f"test time {time.time()-t1:.2f} s")
                if bestscore is None:
                    bestscore = {key: list(results[key]) for key in results}
                for key, result in results.items():
                    writer.add_scalars(f"{key}_{run}", {
                        "trn": result[0],
                        "val": result[1],
                        "tst": result[2]
                    }, epoch)

                if True:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        if valid_hits > bestscore[key][1]:
                            bestscore[key] = list(result)
                            if args.save_gemb:
                                torch.save(h, f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                            if args.savex:
                                torch.save(model.xemb[0].weight.detach(), f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            if args.savemod:
                                torch.save(model.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                                torch.save(predictor.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---', flush=True)
        print(f"best {bestscore}")
        if args.dataset == "collab":
            ret.append(bestscore["Hits@50"][-2:])
        elif args.dataset == "ppa":
            ret.append(bestscore["Hits@100"][-2:])
        elif args.dataset == "ddi":
            ret.append(bestscore["Hits@20"][-2:])
        elif args.dataset == "citation2":
            ret.append(bestscore[-2:])
        elif args.dataset in ["Pubmed", "Cora", "Citeseer"]:
            ret.append(bestscore["Hits@100"][-2:])
        else:
            raise NotImplementedError
    ret = np.array(ret)
    print(ret)
    print(f"Final result: val {np.average(ret[:, 0]):.4f} {np.std(ret[:, 0]):.4f} tst {np.average(ret[:, 1]):.4f} {np.std(ret[:, 1]):.4f}")


if __name__ == "__main__":
    main()