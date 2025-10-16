import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge,GCN2
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
import torch
from torch_sparse import SparseTensor
import torch_sparse
from pygho import SparseTensor as pSparseTensor
from pygho.backend.Spspmm import spsphadamard, spspmm
from pygho.backend.Spmm import spmm
from torch.amp.grad_scaler import GradScaler
import torch
from torch_sparse import SparseTensor
from torch import Tensor
import torch_sparse
from typing import List, Tuple


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float] = [],
          alpha: float = None,
          linkbatchsize: int = None,
          args=None):

    if linkbatchsize is None:
        linkbatchsize = (batch_size + 31) // 32  

    if alpha is not None:
        predictor.setalpha(alpha)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)

    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device),
                                data.adj_t.sizes()[0])
    for perm in PermIterator(adjmask.device, adjmask.shape[0], batch_size):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(
                tei,
                sparse_sizes=(data.num_nodes,
                              data.num_nodes)).to_device(pos_train_edge.device,
                                                         non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        row, col, val = adj.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        loss = 0
        adj = pSparseTensor(torch.stack((row, col)),
                            val,
                            adj.sizes(),
                            is_coalesced=True)
        
        
        h0 = model(data.x, adj)
        h = h0.detach().requires_grad_(True)
        torch.cuda.empty_cache()
        edge = pos_train_edge[:, perm]
        totallen = edge.shape[1]
        for i in range(0, totallen, linkbatchsize):
            start, end = i, min(i + linkbatchsize, totallen)
            tedge = edge[:, start:end]
            cn1,cn2=get_cn1_cn2(adj,tedge)
            pos_outs = predictor.multidomainforward(h,
                                                        adj,
                                                        cn1,
                                                        cn2,
                                                        tedge,
                                                        args,
                                                        cndropprobs=cnprobs)

            pos_losss = -(1 / totallen) * F.logsigmoid(pos_outs).sum()
            
            pos_losss.backward()
            loss += pos_losss.item()
            del pos_outs, pos_losss
        edge = negedge[:, perm]
        totallen = edge.shape[1]
        for i in range(0, totallen, linkbatchsize):
            start, end = i, min(i + linkbatchsize, totallen)
            tedge = edge[:, start:end]

            cn1,cn2=get_cn1_cn2(adj,tedge)
            neg_outs = predictor.multidomainforward(h,
                                                        adj,
                                                        cn1,
                                                        cn2,
                                                        tedge,
                                                        args,
                                                        cndropprobs=cnprobs)
            neg_losss = -(1 / totallen) * F.logsigmoid(-neg_outs).sum()
            
            neg_losss.backward()
            loss += neg_losss.item()
            del neg_outs, neg_losss



        h0.backward(h.grad)
        
        optimizer.step()
        del h, h0
        
        total_loss.append(loss)
    total_loss = np.average([_ for _ in total_loss])
    return total_loss


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


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input,args=None):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    row, col, val = adj.coo()
    if val is None:
        val = torch.ones_like(row, dtype=torch.float)
    adj = pSparseTensor(torch.stack((row, col)), val, adj.sizes(), is_coalesced=True)
    #with torch.autocast("cuda"):
    h = model(data.x, adj)





    pos_valid_pred = torch.cat([
            predictor(h, adj,get_cn1_cn2(adj,pos_valid_edge[perm].t().contiguous())[0],get_cn1_cn2(adj,pos_valid_edge[perm].t().contiguous())[1], pos_valid_edge[perm].t().contiguous(),args).detach().squeeze().cpu()
            for perm in PermIterator(pos_valid_edge.device,
                                    pos_valid_edge.shape[0], batch_size, False)
        ],
                                dim=0)
    neg_valid_pred = torch.cat([
            predictor(h, adj, get_cn1_cn2(adj,neg_valid_edge[perm].t().contiguous())[0],get_cn1_cn2(adj,neg_valid_edge[perm].t().contiguous())[1],neg_valid_edge[perm].t().contiguous(),args).detach().squeeze().cpu()
            for perm in PermIterator(neg_valid_edge.device,
                                    neg_valid_edge.shape[0], batch_size, False)
        ],
                                dim=0)

    if use_valedges_as_input:
        del h
        adj = data.full_adj_t
        row, col, val = adj.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        adj = pSparseTensor(torch.stack((row, col)), val, adj.sizes(), is_coalesced=True)
        h = model(data.x, adj)

    pos_test_pred = torch.cat([
            predictor(h, adj,get_cn1_cn2(adj,pos_test_edge[perm].t().contiguous())[0],get_cn1_cn2(adj,pos_test_edge[perm].t().contiguous())[1], pos_test_edge[perm].t().contiguous(),args).squeeze().cpu()
            for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                    batch_size, False)
        ],
                                dim=0)

    neg_test_pred = torch.cat([
            predictor(h, adj,get_cn1_cn2(adj,neg_test_edge[perm].t().contiguous())[0],get_cn1_cn2(adj,neg_test_edge[perm].t().contiguous())[1], neg_test_edge[perm].t().contiguous(),args).squeeze().cpu()
            for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                    batch_size, False)
        ],
                                dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
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


class Args:
    def __init__(self):
        self.use_valedges_as_input = False
        self.epochs = 150
        self.runs = 2
        self.dataset = "ppa"
        self.batch_size = 16384
        self.linkbatchsize = (self.batch_size+7)//8
        self.testbs = self.linkbatchsize
        self.maskinput = True
        self.mplayers = 1
        self.nnlayers = 3
        self.hiddim = 64
        self.ln = True
        self.lnnn = True
        self.res = False
        self.jk = True
        self.gnndp = 0.0
        self.xdp = 0.0
        self.tdp = 0.0
        self.gnnedp = 0.0
        self.predp = 0.0
        self.preedp = 0.0
        self.gnnlr = 0.0013
        self.prelr = 0.0013
        self.beta = 1
        self.alpha = 1.0
        self.use_xlin = True
        self.tailact = True
        self.twolayerlin = False
        self.increasealpha = False
        self.splitsize = -1
        self.probscale = 4.3
        self.proboffset = 2.8
        self.pt = 0.75
        self.learnpt = False
        self.trndeg = -1
        self.tstdeg = -1
        self.cndeg = -1
        self.predictor = 'cn1'
        self.depth = 1
        self.model = 'gcn'
        self.save_gemb = False
        self.load = None
        self.loadmod = False
        self.savemod = True
        self.savex = False
        self.loadx = False
        self.cnprob = 0
        self.emb_dim = 64
        self.K=5     

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

    if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
        evaluator = Evaluator(name=f'ogbl-ppa')
    else:
        evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load)
    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "cn8", "scn1", "catscn1", "sincn1cn1"]:
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

        
        model = GCN2(data.num_features, args.hiddim, args.hiddim, args.mplayers,
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

        # model, predictor = torch.compile(model), torch.compile(predictor)

        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr},
           {'params': predictor.parameters(), 'lr': args.prelr}], weight_decay=1e-3)

        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, args.maskinput, [], alpha, args.linkbatchsize,args=args)
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