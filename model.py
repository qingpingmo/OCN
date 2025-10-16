from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch import Tensor
import torch
from utils import adjoverlap
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from torch_sparse import SparseTensor
import torch_sparse
from typing import Iterable, Final
import torch
from torch_sparse.tensor import SparseTensor
from torch_sparse import SparseTensor
from torch import Tensor
import torch
from torch_sparse import SparseTensor
from pygho import SparseTensor as pSparseTensor
from pygho.backend.Spspmm import spsphadamard
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


class PureConv(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x
    

convdict = {
    "gcn":
    GCNConv,
    "gcn_cached":
    lambda indim, outdim: GCNConv(indim, outdim, cached=True),
    "sage":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none":
    None
}


class PureConv2(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn", use_lin=False) -> None:
        super().__init__()
        self.aggr = aggr
        if not use_lin:
            if indim == outdim:
                self.lin = nn.Identity()
            else:
                raise NotImplementedError
        else:
            self.lin = nn.Sequential(nn.Linear(indim, outdim, bias=False), nn.ReLU(inplace=True))

    def forward(self, x, adj_t: pSparseTensor):
        if self.aggr == "mean":
            x= spmm(adj_t.tuplewiseapply(lambda ea: ea.unsqueeze(-1)), 1, x, aggr="mean")
        elif self.aggr == "max":
            x= spmm(adj_t.tuplewiseapply(lambda ea: ea.unsqueeze(-1)), 1, x, aggr="amax")
        elif self.aggr == "sum":
            x= adj_t.to_torch_sparse_coo()@x
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dims=1)))
            enorm = norm[adj_t.indices[0]]*norm[adj_t.indices[1]]
            
            normed_adj = adj_t.tuplewiseapply(lambda ea: (ea*enorm))
            with torch.autocast(device_type="cuda", enabled=False):
                x = normed_adj.to_torch_sparse_coo()@x #
            x = x
        return self.lin(x)
    
class PureConv3(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn", use_lin=False) -> None:
        super().__init__()
        self.aggr = aggr
        if not use_lin:
            if indim == outdim:
                self.lin = nn.Identity()
            else:
                raise NotImplementedError
        else:
            self.lin = nn.Sequential(nn.Linear(indim, outdim, bias=False), nn.ReLU(inplace=True))

    def forward(self, x, adj_t: pSparseTensor):
        if self.aggr == "mean":
            x= spmm(adj_t.tuplewiseapply(lambda ea: ea.unsqueeze(-1)), 1, x, aggr="mean")
        elif self.aggr == "max":
            x= spmm(adj_t.tuplewiseapply(lambda ea: ea.unsqueeze(-1)), 1, x, aggr="amax")
        elif self.aggr == "sum":
            x= adj_t.to_torch_sparse_coo()@x
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dims=1)))
            enorm = norm[adj_t.indices[0]]*norm[adj_t.indices[1]]
            
            normed_adj = adj_t.tuplewiseapply(lambda ea: (ea*enorm))
            x = normed_adj.to_torch_sparse_coo()@x 
            x = x
        return self.lin(x)



convdict2 = {
    "gcn":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="gcn", use_lin=True),
    "gcn_cached":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="gcn", use_lin=True),
    "sage":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="mean", use_lin=True),
    "gin":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="sum", use_lin=True),
    "max":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="max", use_lin=True),
    "puremax":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="max"),
    "puresum":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="sum"),
    "puremean":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="mean"),
    "puregcn":
    lambda indim, outdim: PureConv2(indim, outdim, aggr="gcn"),
    "none":
    None
}


convdict3 = {
    "gcn":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="gcn", use_lin=True),
    "gcn_cached":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="gcn", use_lin=True),
    "sage":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="mean", use_lin=True),
    "gin":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="sum", use_lin=True),
    "max":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="max", use_lin=True),
    "puremax":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="max"),
    "puresum":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="sum"),
    "puremean":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="mean"),
    "puregcn":
    lambda indim, outdim: PureConv3(indim, outdim, aggr="gcn"),
    "none":
    None
}

predictor_dict = {}




class DropEdge(nn.Module):

    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index: Tensor):
        if self.dp == 0:
            return edge_index
        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp
        return edge_index[:, mask]


class DropAdj(nn.Module):
    doscale: Final[bool] 
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj


class GCN(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False):
        super().__init__()
        
        self.adjdrop = DropAdj(edrop)
        
        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            if not noinputlin and ("pure" in conv_fn or num_layers==0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
            
        if num_layers == 0 or conv_fn =="none":
            self.jk = False
            return
        
        convfn = convdict[conv_fn]
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers-1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                            nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())
        

    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, self.adjdrop(adj_t)))
            if self.res and x1.shape[-1] == x.shape[-1]: # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk: # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        return x


class GCN2(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False):
        super().__init__()

        self.adjdrop = DropAdj(edrop)

        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            if not noinputlin and ("pure" in conv_fn or num_layers==0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())

        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))

        if num_layers == 0 or conv_fn =="none":
            self.jk = False
            return

        convfn = convdict2[conv_fn]
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers-1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                            nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())


    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, adj_t))
            if self.res and x1.shape[-1] == x.shape[-1]: # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk: # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        return x


class GCN3(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False):
        super().__init__()

        self.adjdrop = DropAdj(edrop)

        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            if not noinputlin and ("pure" in conv_fn or num_layers==0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())

        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))

        if num_layers == 0 or conv_fn =="none":
            self.jk = False
            return

        convfn = convdict3[conv_fn]
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers-1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                            nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())


    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, adj_t))
            if self.res and x1.shape[-1] == x.shape[-1]: 
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk: 
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        return x



def sparse_identity(n):
    row = torch.arange(n)
    col = torch.arange(n)
    value = torch.ones(n)
    
    sparse_I = SparseTensor(row=row, col=col, value=value, sparse_sizes=(n, n))
    return sparse_I


class CNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        
        
        self.xijlin = nn.Sequential(
            nn.Linear(64, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_buffer("innerprod", torch.tensor([0.0]))
        self.n = 0

    def innerprod1(self, E1, E2):
        if self.training:
            hprod = spsphadamard(E1, E2)
            innerprod = hprod.values.sum()
            self.n += 1
            beta = self.n**-1
            self.innerprod *= (1 - beta)
            self.innerprod += beta * innerprod
        # print("innerprod", self.innerprod)
        return self.innerprod

    def multidomainforward(self,
                           x,
                           adj,
                           cn1,
                           cn2,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        
        
        col_sum = cn1.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        inv_col_sum[~non_empty_cols] = 0 

        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn1 = cn1.mul(inv_col_sum)  
        
        num_rows = normalized_cn1.size(0) 
        
        num_cols = normalized_cn1.size(1)  

        Ortho_zero = SparseTensor(
            row=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            col=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            value=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            sparse_sizes=(num_rows, num_cols)
        )



        spnormalized_cn1=normalized_cn1.to_torch_sparse_coo_tensor()
        spOrtho_zero=Ortho_zero.to_torch_sparse_coo_tensor()
        

        spcn2=cn2.to_torch_sparse_coo_tensor()



        spcn2_copy=spcn2
        spnormalized_cn1_copy=spnormalized_cn1        
        spcn2 = spcn2.coalesce()
        spnormalized_cn1 = spnormalized_cn1.coalesce()


        num_nodes = spnormalized_cn1.shape[1] 
        num_edges = spnormalized_cn1.shape[0]

        row1, col1 = spnormalized_cn1.indices()
        row2, col2 = spcn2.indices()

 
        value1 = spnormalized_cn1.values()
        value2 = spcn2.values()



        spnormalized_cn1 = torch_sparse.SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=(num_edges, num_nodes))
        spcn2 = torch_sparse.SparseTensor(row=row2, col=col2, value=value2, sparse_sizes=(num_edges, num_nodes))



        row, col, val = spcn2.coo()
        spcn2 = pSparseTensor(torch.stack((row, col)), val, spcn2.sizes(), is_coalesced=True)

        row, col, val = spnormalized_cn1.coo()
        spnormalized_cn1 = pSparseTensor(torch.stack((row, col)), val, spnormalized_cn1.sizes(), is_coalesced=True)


        inner_product = self.innerprod1(spcn2, spnormalized_cn1)

           



        spcn2 = spcn2_copy.coalesce()
        spnormalized_cn1 = spnormalized_cn1_copy.coalesce()
        spOrtho_zero = spOrtho_zero.coalesce()

           
           
            
        spcn2_indices = spcn2.indices()
        spnormalized_cn1_indices = spnormalized_cn1.indices()
        spOrtho_zero_indices = spOrtho_zero.indices()

        spcn2_values = spcn2.values()
        spnormalized_cn1_values = spnormalized_cn1.values()
        spOrtho_zero_values = spOrtho_zero.values()

            
        unique_indices, inverse_indices = torch.unique(
                torch.cat((spcn2_indices, spnormalized_cn1_indices, spOrtho_zero_indices), dim=1),
                dim=1,
                return_inverse=True
            )

        spcn2_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spcn2_aligned_values[inverse_indices[:spcn2_indices.size(1)]] = spcn2_values

        spnormalized_cn1_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spnormalized_cn1_aligned_values[inverse_indices[spcn2_indices.size(1):spcn2_indices.size(1) + spnormalized_cn1_indices.size(1)]] = spnormalized_cn1_values

        spOrtho_zero_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        if spOrtho_zero._nnz() > 0:
            spOrtho_zero_aligned_values[inverse_indices[spcn2_indices.size(1) + spnormalized_cn1_indices.size(1):]] = spOrtho_zero_values

            
            
        if spnormalized_cn1_aligned_values.numel() > 0:
                
            scale_factor = spnormalized_cn1_aligned_values.abs().max().item()
        else:
        
            scale_factor = 1.0  
        normalized_inner_product = inner_product / scale_factor if scale_factor > 0 else inner_product



        new_values = (
                spcn2_aligned_values
                - normalized_inner_product * spnormalized_cn1_aligned_values
                
            )
            
        spcn2 = torch.sparse_coo_tensor(
                unique_indices,
                new_values,
                size=spcn2.size(),
                device=spcn2.device
            ).coalesce()





            
        spcn2 = spcn2.coalesce()

            
        indices = spcn2.indices()
        values = spcn2.values()

            
        col_sum = torch.zeros(spcn2.size(1), device=spcn2.device)
        col_sum.index_add_(0, indices[1], values)  

            
        col_sum[col_sum == 0] = 1
        inv_col_sum = 1 / col_sum

            
        normalized_values = values * inv_col_sum[indices[1]]

            
        row, col = indices
        normalized_spcn2_sparse = SparseTensor(
                row=row,
                col=col,
                value=normalized_values,
                sparse_sizes=spcn2.size(),
                is_sorted=True  
            )

        n = normalized_cn1.size(1)  
        dev = normalized_cn1.device() 

        indices = torch.arange(n, device=normalized_cn1.device()).unsqueeze(0)
        indices = torch.cat([indices, indices], dim=0)  
        values = torch.ones(n, device=normalized_cn1.device())  

        
        sparse_identity_matrix = torch.sparse_coo_tensor(indices, values, (n, n), device=normalized_cn1.device())
        
        sparse_identity_matrix = sparse_identity_matrix.coalesce()

        um_nodes = sparse_identity_matrix.shape[1] 
        num_edges = sparse_identity_matrix.shape[0]

        row1, col1 = sparse_identity_matrix.indices()
        

 
        value1 = sparse_identity_matrix.values()
        

        sparse_identity_matrix = torch_sparse.SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=(num_edges, num_nodes))

        row, col, val = sparse_identity_matrix.coo()
        sparse_identity_matrix = pSparseTensor(torch.stack((row, col)), val, sparse_identity_matrix.sizes(), is_coalesced=True)



        row, col, val = normalized_cn1.coo()
        normalized_cn1 = pSparseTensor(torch.stack((row, col)), val, normalized_cn1.sizes(), is_coalesced=True)

        row, col, val = normalized_spcn2_sparse.coo()
        normalized_spcn2_sparse = pSparseTensor(torch.stack((row, col)), val, normalized_spcn2_sparse.sizes(), is_coalesced=True)

        xcn1=spspmm(normalized_cn1, 1, sparse_identity_matrix, 0)
        xcn2=spspmm(normalized_spcn2_sparse, 1, sparse_identity_matrix, 0)


        xcn1=xcn1.to_torch_sparse_coo()
        xcn2=xcn2.to_torch_sparse_coo()
        xcn1=xcn1.coalesce()
        xcn2=xcn2.coalesce()
        num_edge=xcn1.size(0)
        num_node=xcn1.size(1)
        indices1 = xcn1.indices()
        indices2= xcn2.indices()
        values1 = xcn1.values()
        values2 = xcn2.values()
        xcn1 = torch.sparse_coo_tensor(indices1, values1, (num_edge, num_node), device=dev)
        xcn2 = torch.sparse_coo_tensor(indices2, values2, (num_edge, num_node), device=dev)


        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        xcn1 = self.xcn1lin(xcn1)
        
        xcn2 = self.xcn2lin(xcn2)



        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 + self.beta * xij)
        
        
        return x
    
    def forward(self, x, adj,cn1,cn2, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj,cn1,cn2, tar_ei, filled1, [])


class IncompleteCN1Predictor(CNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=8,
                 testresdeg=128,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1,
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin, tailact, twolayerlin, beta)
        self.learnablept= learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha2", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))

        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1), nn.Sigmoid())

    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale*(prob-self.offset))
        return self.alpha2*pt*p0/(pt*p0+1-p0)

    def multidomainforward(self,
                           x,
                           adj,
                           
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int=None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        
        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        x = x + self.xlin(x)
        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        else:
            cn = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=False,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        xcns = [spmm_add(cn, x)]
        
        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, ei1,
                        filled1, depth-1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, ei2,
                        filled1, depth-1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i+self.splitsize] = self.forward(x, adj,ei1[:, i: i+self.splitsize], filled1, depth-1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i+self.splitsize] = self.forward(x, adj, ei2[:, i: i+self.splitsize],filled1, depth-1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]]) 
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")


            col_sum = cnres1.sum(dim=0)

            col_sum[col_sum == 0] = 1  
            inv_col_sum = 1 / col_sum
            non_empty_cols = col_sum != 1  
            inv_col_sum[~non_empty_cols] = 0  

            inv_col_sum = inv_col_sum.view(1, -1)  

            
            normalized_cn1 = cnres1.mul(inv_col_sum)  
            
            num_rows = normalized_cn1.size(0) 
            
            num_cols = normalized_cn1.size(1)  

            Ortho_zero = SparseTensor(
                row=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
                col=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
                value=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
                sparse_sizes=(num_rows, num_cols)
            )



            spnormalized_cn1=normalized_cn1.to_torch_sparse_coo_tensor()
            spOrtho_zero=Ortho_zero.to_torch_sparse_coo_tensor()
            


            spcn2=cnres2.to_torch_sparse_coo_tensor()



            spcn2_copy=spcn2
            spnormalized_cn1_copy=spnormalized_cn1        
            spcn2 = spcn2.coalesce()
            spnormalized_cn1 = spnormalized_cn1.coalesce()


            num_nodes = spnormalized_cn1.shape[1] 
            num_edges = spnormalized_cn1.shape[0]

            row1, col1 = spnormalized_cn1.indices()
            row2, col2 = spcn2.indices()

    
            value1 = spnormalized_cn1.values()
            value2 = spcn2.values()



            spnormalized_cn1 = torch_sparse.SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=(num_edges, num_nodes))
            spcn2 = torch_sparse.SparseTensor(row=row2, col=col2, value=value2, sparse_sizes=(num_edges, num_nodes))



            row, col, val = spcn2.coo()
            spcn2 = pSparseTensor(torch.stack((row, col)), val, spcn2.sizes(), is_coalesced=True)

            row, col, val = spnormalized_cn1.coo()
            spnormalized_cn1 = pSparseTensor(torch.stack((row, col)), val, spnormalized_cn1.sizes(), is_coalesced=True)


            inner_product = self.innerprod1(spcn2, spnormalized_cn1)

            



            spcn2 = spcn2_copy.coalesce()
            spnormalized_cn1 = spnormalized_cn1_copy.coalesce()
            spOrtho_zero = spOrtho_zero.coalesce()

            
            
                
            spcn2_indices = spcn2.indices()
            spnormalized_cn1_indices = spnormalized_cn1.indices()
            spOrtho_zero_indices = spOrtho_zero.indices()

            spcn2_values = spcn2.values()
            spnormalized_cn1_values = spnormalized_cn1.values()
            spOrtho_zero_values = spOrtho_zero.values()

                
            unique_indices, inverse_indices = torch.unique(
                    torch.cat((spcn2_indices, spnormalized_cn1_indices, spOrtho_zero_indices), dim=1),
                    dim=1,
                    return_inverse=True
                )

            spcn2_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
            spcn2_aligned_values[inverse_indices[:spcn2_indices.size(1)]] = spcn2_values

            spnormalized_cn1_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
            spnormalized_cn1_aligned_values[inverse_indices[spcn2_indices.size(1):spcn2_indices.size(1) + spnormalized_cn1_indices.size(1)]] = spnormalized_cn1_values

            spOrtho_zero_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
            if spOrtho_zero._nnz() > 0:
                spOrtho_zero_aligned_values[inverse_indices[spcn2_indices.size(1) + spnormalized_cn1_indices.size(1):]] = spOrtho_zero_values

                
                
            if spnormalized_cn1_aligned_values.numel() > 0:
                    
                scale_factor = spnormalized_cn1_aligned_values.abs().max().item()
            else:
            
                scale_factor = 1.0  
            normalized_inner_product = inner_product / scale_factor if scale_factor > 0 else inner_product



            new_values = (
                    spcn2_aligned_values
                    - normalized_inner_product * spnormalized_cn1_aligned_values
                    
                )
                
            spcn2 = torch.sparse_coo_tensor(
                    unique_indices,
                    new_values,
                    size=spcn2.size(),
                    device=spcn2.device
                ).coalesce()





                
            spcn2 = spcn2.coalesce()

                
            indices = spcn2.indices()
            values = spcn2.values()

                
            col_sum = torch.zeros(spcn2.size(1), device=spcn2.device)
            col_sum.index_add_(0, indices[1], values)  

                
            col_sum[col_sum == 0] = 1
            inv_col_sum = 1 / col_sum

                
            normalized_values = values * inv_col_sum[indices[1]]

                
            row, col = indices
            normalized_spcn2_sparse = SparseTensor(
                    row=row,
                    col=col,
                    value=normalized_values,
                    sparse_sizes=spcn2.size(),
                    is_sorted=True  
                )




            xcn1 = spmm_add(normalized_cn1, x)
            xcn2 = spmm_add(normalized_spcn2_sparse, x)
            xcns[0] = xcns[0] + xcn2 + xcn1
        
        xij = self.xijlin(xij)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def setalpha(self, alpha: float):
        self.alpha2.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj, tar_ei, filled1, [],
                                       depth)



class IncompleteCN1Predictorhighorder(CNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=8,
                 testresdeg=128,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1,
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin, tailact, twolayerlin, beta)
        self.learnablept= learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha2", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))

        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1), nn.Sigmoid())

    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale*(prob-self.offset))
        return self.alpha2*pt*p0/(pt*p0+1-p0)

    def multidomainforward(self,
                           x,
                           adj,
                           cn1,
                           cn2,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int=None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        
        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        x = x + self.xlin(x)
        spadj = adj.to_torch_sparse_coo_tensor()
        adj2 = SparseTensor.from_torch_sparse_coo_tensor(spadj @ spadj, False)
        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
            
            cn22, cn2res1, cn2res2 =adjoverlap(adj, adj2, tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)


        else:
            cn = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=False,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
            
            cn22,_,_=adjoverlap(adj, adj2, tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)


        col_sum = cn.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        
        inv_col_sum[~non_empty_cols] = 1

        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn1 = cn.mul(inv_col_sum)  
        
        num_rows = normalized_cn1.size(0) 
        
        num_cols = normalized_cn1.size(1)  

        Ortho_zero = SparseTensor(
            row=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            col=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            value=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            sparse_sizes=(num_rows, num_cols)
        )



        spnormalized_cn1=normalized_cn1.to_torch_sparse_coo_tensor()
        spOrtho_zero=Ortho_zero.to_torch_sparse_coo_tensor()
        




        

        spcn2=cn22.to_torch_sparse_coo_tensor()



        spcn2_copy=spcn2
        spnormalized_cn1_copy=spnormalized_cn1        
        spcn2 = spcn2.coalesce()
        spnormalized_cn1 = spnormalized_cn1.coalesce()


        num_nodes = spnormalized_cn1.shape[1] 
        num_edges = spnormalized_cn1.shape[0]

        row1, col1 = spnormalized_cn1.indices()
        row2, col2 = spcn2.indices()

 
        value1 = spnormalized_cn1.values()
        value2 = spcn2.values()



        spnormalized_cn1 = torch_sparse.SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=(num_edges, num_nodes))
        spcn2 = torch_sparse.SparseTensor(row=row2, col=col2, value=value2, sparse_sizes=(num_edges, num_nodes))



        row, col, val = spcn2.coo()
        spcn2 = pSparseTensor(torch.stack((row, col)), val, spcn2.sizes(), is_coalesced=True)

        row, col, val = spnormalized_cn1.coo()
        spnormalized_cn1 = pSparseTensor(torch.stack((row, col)), val, spnormalized_cn1.sizes(), is_coalesced=True)


        inner_product = self.innerprod1(spcn2, spnormalized_cn1)

           



        spcn2 = spcn2_copy.coalesce()
        spnormalized_cn1 = spnormalized_cn1_copy.coalesce()
        spOrtho_zero = spOrtho_zero.coalesce()

           
           
            
        spcn2_indices = spcn2.indices()
        spnormalized_cn1_indices = spnormalized_cn1.indices()
        spOrtho_zero_indices = spOrtho_zero.indices()

        spcn2_values = spcn2.values()
        spnormalized_cn1_values = spnormalized_cn1.values()
        spOrtho_zero_values = spOrtho_zero.values()

            
        unique_indices, inverse_indices = torch.unique(
                torch.cat((spcn2_indices, spnormalized_cn1_indices, spOrtho_zero_indices), dim=1),
                dim=1,
                return_inverse=True
            )

        spcn2_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spcn2_aligned_values[inverse_indices[:spcn2_indices.size(1)]] = spcn2_values

        spnormalized_cn1_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spnormalized_cn1_aligned_values[inverse_indices[spcn2_indices.size(1):spcn2_indices.size(1) + spnormalized_cn1_indices.size(1)]] = spnormalized_cn1_values

        spOrtho_zero_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        if spOrtho_zero._nnz() > 0:
            spOrtho_zero_aligned_values[inverse_indices[spcn2_indices.size(1) + spnormalized_cn1_indices.size(1):]] = spOrtho_zero_values

            
            
        if spnormalized_cn1_aligned_values.numel() > 0:
                
            scale_factor = spnormalized_cn1_aligned_values.abs().max().item()
        else:
        
            scale_factor = 1.0  
        normalized_inner_product = inner_product / scale_factor if scale_factor > 0 else inner_product



        new_values = (
                spcn2_aligned_values
                - normalized_inner_product * spnormalized_cn1_aligned_values
                
            )
            
        spcn2 = torch.sparse_coo_tensor(
                unique_indices,
                new_values,
                size=spcn2.size(),
                device=spcn2.device
            ).coalesce()





            
        spcn2 = spcn2.coalesce()

            
        indices = spcn2.indices()
        values = spcn2.values()

            
        col_sum = torch.zeros(spcn2.size(1), device=spcn2.device)
        col_sum.index_add_(0, indices[1], values)  

            
        col_sum[col_sum == 0] = 1
        inv_col_sum = 1 / col_sum

            
        normalized_values = values * inv_col_sum[indices[1]]

            
        row, col = indices
        normalized_spcn2_sparse = SparseTensor(
                row=row,
                col=col,
                value=normalized_values,
                sparse_sizes=spcn2.size(),
                is_sorted=True  
            )


        xcns = [spmm_add(normalized_cn1, x)]
        xcns2=[spmm_add(normalized_spcn2_sparse, x)]

        
        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, cn1,cn2,ei1,
                        filled1, depth-1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, cn1,cn2,ei2,
                        filled1, depth-1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i+self.splitsize] = self.forward(x, adj,cn1,cn2, ei1[:, i: i+self.splitsize], filled1, depth-1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i+self.splitsize] = self.forward(x, adj,cn1,cn2, ei2[:, i: i+self.splitsize],filled1, depth-1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]]) 
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")
            xcn1 = spmm_add(cnres1, x)
            xcn2 = spmm_add(cnres2, x)
            xcns[0] = xcns[0] + xcn2 + xcn1


        if depth > 0.5:
            potcn1 = cn2res1.coo()
            potcn2 = cn2res2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, cn1,cn2,ei1,
                        filled1, depth-1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, cn1,cn2,ei2,
                        filled1, depth-1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i+self.splitsize] = self.forward(x, adj, cn1,cn2,ei1[:, i: i+self.splitsize], filled1, depth-1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i+self.splitsize] = self.forward(x, adj, cn1,cn2,ei2[:, i: i+self.splitsize],filled1, depth-1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]]) 
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cn2res1.set_value_(probcn1, layout="coo")
            cn2res2.set_value_(probcn2, layout="coo")
            xcn1 = spmm_add(cn2res1, x)
            xcn2 = spmm_add(cn2res2, x)
            xcns2[0] = xcns2[0] + xcn2 + xcn1

        xij = self.xijlin(xij)
        
        xs = torch.cat(
            [
                self.lin(self.xcnlin(xcn) * self.beta + self.xcnlin(xcn2) * self.beta + xij)
                for xcn in xcns
                for xcn2 in xcns2
            ],
            dim=-1
        )
        return xs

    def setalpha(self, alpha: float):
        self.alpha2.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                cn1,
                cn2,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj,cn1,cn2, tar_ei, filled1, [],
                                       depth)




import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from typing import Final, Iterable

class IncompleteCN1PredictorSaveMemory(CNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=8,
                 testresdeg=128,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin, tailact, twolayerlin, beta)
        self.learnablept = learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha2", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))
        
        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale * (prob - self.offset))
        result = self.alpha2 * pt * p0 / (pt * p0 + 1 - p0)
        del prob, p0  
        torch.cuda.empty_cache()
        return result
    
    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int = None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        del cndropprobs 

        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        
        x = x + self.xlin(x)
         

        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                adj,
                adj,
                tar_ei,
                filled1,
                calresadj=True,
                cnsampledeg=self.cndeg,
                ressampledeg=self.trainresdeg if self.training else self.testresdeg
            )
        else:
            cn = adjoverlap(
                adj,
                adj,
                tar_ei,
                filled1,
                calresadj=False,
                cnsampledeg=self.cndeg,
                ressampledeg=self.trainresdeg if self.training else self.testresdeg
            )
            del adj  

        xcns = [spmm_add(cn, x)]
        del cn  

        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            

            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    

                    probcn1 = self.forward(
                        x, adj, ei1,
                        filled1, depth - 1
                    ).flatten()
                    del ei1  

                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, ei2,
                        filled1, depth - 1
                    ).flatten()
                    del ei2  
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    del tar_ei  

                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i + self.splitsize] = self.forward(
                            x, adj, ei1[:, i: i + self.splitsize],
                            filled1, depth - 1
                        ).flatten()
                        del i  

                    del ei1, num1  

                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i + self.splitsize] = self.forward(
                            x, adj, ei2[:, i: i + self.splitsize],
                            filled1, depth - 1
                        ).flatten()
                        del i 

                    del ei2, num2  

            if self.learnablept:
                pt = self.ptlin(xij)
                  
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]])
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
                del pt  
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
                

            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            del potcn1, potcn2  

            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")
            del probcn1, probcn2  

            col_sum = cnres1.sum(dim=0)
            

            col_sum[col_sum == 0] = 1
            inv_col_sum = 1 / col_sum
            del col_sum  

            non_empty_cols = inv_col_sum != 1
            inv_col_sum[~non_empty_cols] = 1
            del non_empty_cols 

            inv_col_sum = inv_col_sum.view(1, -1)

            normalized_cn1 = cnres1.mul(inv_col_sum)
            del inv_col_sum  

            
            Ortho_zero = SparseTensor(
                row=torch.tensor([], dtype=torch.long, device=normalized_cn1.device()),
                col=torch.tensor([], dtype=torch.long, device=normalized_cn1.device()),
                value=torch.tensor([], dtype=torch.float, device=normalized_cn1.device()),
                sparse_sizes=(normalized_cn1.size(0), normalized_cn1.size(1))
            )
            

            spnormalized_cn1 = normalized_cn1.to_torch_sparse_coo_tensor()
            spOrtho_zero = Ortho_zero.to_torch_sparse_coo_tensor()
            del Ortho_zero  

            spcn2 = cnres2.to_torch_sparse_coo_tensor()
            del cnres2  

            spcn2_copy = spcn2
            spnormalized_cn1_copy = spnormalized_cn1
            spcn2 = spcn2.coalesce()
            spnormalized_cn1 = spnormalized_cn1.coalesce()

            num_nodes = spnormalized_cn1.size(1)
            num_edges = spnormalized_cn1.size(0)
            

            row1, col1 = spnormalized_cn1.indices()
            row2, col2 = spcn2.indices()
            

            value1 = spnormalized_cn1.values()
            value2 = spcn2.values()
            del spcn2  

            spnormalized_cn1 = torch_sparse.SparseTensor(
                row=row1, col=col1, value=value1,
                sparse_sizes=(num_edges, num_nodes)
            )
            spcn2 = torch_sparse.SparseTensor(
                row=row2, col=col2, value=value2,
                sparse_sizes=(num_edges, num_nodes)
            )
            del row1, col1, value1, row2, col2, value2  

            row, col, val = spcn2.coo()
            spcn2 = pSparseTensor(torch.stack((row, col)), val, spcn2.sizes(), is_coalesced=True)
            del row, col, val  

            row, col, val = spnormalized_cn1.coo()
            spnormalized_cn1 = pSparseTensor(torch.stack((row, col)), val, spnormalized_cn1.sizes(), is_coalesced=True)
            del row, col, val  

            inner_product = self.innerprod1(spcn2, spnormalized_cn1)
            del spnormalized_cn1, spcn2 

            spcn2 = spcn2_copy.coalesce()
            spnormalized_cn1 = spnormalized_cn1_copy.coalesce()
            spOrtho_zero = spOrtho_zero.coalesce()
            del spcn2_copy, spnormalized_cn1_copy  

            spcn2_indices = spcn2.indices()
            spnormalized_cn1_indices = spnormalized_cn1.indices()
            spOrtho_zero_indices = spOrtho_zero.indices()
            
            spcn2_values = spcn2.values()
            spnormalized_cn1_values = spnormalized_cn1.values()
            spOrtho_zero_values = spOrtho_zero.values()
            del  spnormalized_cn1  

            unique_indices, inverse_indices = torch.unique(
                torch.cat((spcn2_indices, spnormalized_cn1_indices, spOrtho_zero_indices), dim=1),
                dim=1,
                return_inverse=True
            )
            del spcn2_indices, spnormalized_cn1_indices, spOrtho_zero_indices  

            spcn2_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
            spcn2_aligned_values[inverse_indices[:spcn2_values.size(0)]] = spcn2_values
            

            spnormalized_cn1_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
            spnormalized_cn1_aligned_values[inverse_indices[spcn2_values.size(0):spcn2_values.size(0) + spnormalized_cn1_values.size(0)]] = spnormalized_cn1_values
            del spnormalized_cn1_values  

            spOrtho_zero_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
            if spOrtho_zero_values.numel() > 0:
                spOrtho_zero_aligned_values[inverse_indices[spcn2_values.size(0) + spnormalized_cn1_values.size(0):]] = spOrtho_zero_values
            del spOrtho_zero_values, inverse_indices 

            if spnormalized_cn1_aligned_values.numel() > 0:
                scale_factor = spnormalized_cn1_aligned_values.abs().max().item()
            else:
                scale_factor = 1.0
            

            normalized_inner_product = inner_product / scale_factor if scale_factor > 0 else inner_product
            del scale_factor  

            new_values = (
                spcn2_aligned_values
                - normalized_inner_product * spnormalized_cn1_aligned_values
            )
            del spcn2_aligned_values, normalized_inner_product 

            spcn2 = torch.sparse_coo_tensor(
                unique_indices,
                new_values,
                size=spcn2.size(),
                device=spcn2.device
            ).coalesce()
            del unique_indices, new_values  

            spcn2 = spcn2.coalesce()
            

            indices = spcn2.indices()
            values = spcn2.values()
            

            col_sum = torch.zeros(spcn2.size(1), device=spcn2.device)
            col_sum.index_add_(0, indices[1], values)
            

            col_sum[col_sum == 0] = 1
            inv_col_sum = 1 / col_sum
            del col_sum  

            normalized_values = values * inv_col_sum[indices[1]]
            del inv_col_sum  

            row, col = indices
            del indices  

            normalized_spcn2_sparse = SparseTensor(
                row=row,
                col=col,
                value=normalized_values,
                sparse_sizes=spcn2.size(),
                is_sorted=True
            )
            del row, col, normalized_values  

            xcn1 = spmm_add(normalized_cn1, x)
            del normalized_cn1  
            xcn2 = spmm_add(normalized_spcn2_sparse, x)
            del normalized_spcn2_sparse 
            del x  

            xcns[0] = xcns[0] + xcn2 + xcn1
            del xcn1, xcn2  

        xij = self.xijlin(xij)
        

        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + self.xijlin(xij)) for xcn in xcns],
            dim=-1
        )
        del xcns  
        del xij  

        return xs

    def setalpha(self, alpha: float):
        self.alpha2.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj, tar_ei, filled1, [], depth)






class CNLinkPredictorbasis(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        
        
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_buffer("innerprod", torch.tensor([0.0]))
        self.n = 0

    def innerprod1(self, E1, E2):
        if self.training:
            hprod = spsphadamard(E1, E2)
            innerprod = hprod.values.sum()
            self.n += 1
            beta = self.n**-1
            self.innerprod *= (1 - beta)
            self.innerprod += beta * innerprod
        # print("innerprod", self.innerprod)
        return self.innerprod

    def multidomainforward(self,
                           x,
                           adj,
                           cn1,
                           cn2,
                           tar_ei,
                           args,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []
                           ):
        
      
        col_sum = cn1.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
       
        inv_col_sum[~non_empty_cols] = args.sum  


        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn1 = cn1.mul(inv_col_sum)  
        
        num_rows = normalized_cn1.size(0) 
        
        num_cols = normalized_cn1.size(1)  

        Ortho_zero = SparseTensor(
            row=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            col=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            value=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            sparse_sizes=(num_rows, num_cols)
        )



        spnormalized_cn1=normalized_cn1.to_torch_sparse_coo_tensor()
        spOrtho_zero=Ortho_zero.to_torch_sparse_coo_tensor()
        




       

        spcn2=cn2.to_torch_sparse_coo_tensor()



        spcn2_copy=spcn2
        spnormalized_cn1_copy=spnormalized_cn1        
        spcn2 = spcn2.coalesce()
        spnormalized_cn1 = spnormalized_cn1.coalesce()


        num_nodes = spnormalized_cn1.shape[1] 
        num_edges = spnormalized_cn1.shape[0]

        row1, col1 = spnormalized_cn1.indices()
        row2, col2 = spcn2.indices()

 
        value1 = spnormalized_cn1.values()
        value2 = spcn2.values()



        spnormalized_cn1 = torch_sparse.SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=(num_edges, num_nodes))
        spcn2 = torch_sparse.SparseTensor(row=row2, col=col2, value=value2, sparse_sizes=(num_edges, num_nodes))



        row, col, val = spcn2.coo()
        spcn2 = pSparseTensor(torch.stack((row, col)), val, spcn2.sizes(), is_coalesced=True)

        row, col, val = spnormalized_cn1.coo()
        spnormalized_cn1 = pSparseTensor(torch.stack((row, col)), val, spnormalized_cn1.sizes(), is_coalesced=True)


        inner_product = self.innerprod1(spcn2, spnormalized_cn1)

           



        spcn2 = spcn2_copy.coalesce()
        spnormalized_cn1 = spnormalized_cn1_copy.coalesce()
        spOrtho_zero = spOrtho_zero.coalesce()

           
           
            
        spcn2_indices = spcn2.indices()
        spnormalized_cn1_indices = spnormalized_cn1.indices()
        spOrtho_zero_indices = spOrtho_zero.indices()

        spcn2_values = spcn2.values()
        spnormalized_cn1_values = spnormalized_cn1.values()
        spOrtho_zero_values = spOrtho_zero.values()

            
        unique_indices, inverse_indices = torch.unique(
                torch.cat((spcn2_indices, spnormalized_cn1_indices, spOrtho_zero_indices), dim=1),
                dim=1,
                return_inverse=True
            )

        spcn2_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spcn2_aligned_values[inverse_indices[:spcn2_indices.size(1)]] = spcn2_values

        spnormalized_cn1_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spnormalized_cn1_aligned_values[inverse_indices[spcn2_indices.size(1):spcn2_indices.size(1) + spnormalized_cn1_indices.size(1)]] = spnormalized_cn1_values

        spOrtho_zero_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        if spOrtho_zero._nnz() > 0:
            spOrtho_zero_aligned_values[inverse_indices[spcn2_indices.size(1) + spnormalized_cn1_indices.size(1):]] = spOrtho_zero_values

            
            
        if spnormalized_cn1_aligned_values.numel() > 0:
                
            scale_factor = spnormalized_cn1_aligned_values.abs().max().item()
        else:
        
            scale_factor = 1.0  
        normalized_inner_product = inner_product / scale_factor if scale_factor > 0 else inner_product



        new_values = (
                spcn2_aligned_values
                - normalized_inner_product * spnormalized_cn1_aligned_values
                
            )
            
        spcn2 = torch.sparse_coo_tensor(
                unique_indices,
                new_values,
                size=spcn2.size(),
                device=spcn2.device
            ).coalesce()





            
        spcn2 = spcn2.coalesce()

            
        indices = spcn2.indices()
        values = spcn2.values()

            
        col_sum = torch.zeros(spcn2.size(1), device=spcn2.device)
        col_sum.index_add_(0, indices[1], values)  

            
        col_sum[col_sum == 0] = 1
        inv_col_sum = 1 / col_sum

            
        normalized_values = values * inv_col_sum[indices[1]]

            
        row, col = indices
        normalized_spcn2_sparse = SparseTensor(
                row=row,
                col=col,
                value=normalized_values,
                sparse_sizes=spcn2.size(),
                is_sorted=True  
            )

           
        xcn1=spmm_add(normalized_cn1, x)
        xcn2=spmm_add(normalized_spcn2_sparse, x)

        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        
        xcn1 = self.xcn1lin(xcn1)
        
        xcn2 = self.xcn2lin(xcn2)

        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 + self.beta * xij)
        
        
        return x
    
    def forward(self, x, adj,cn1,cn2, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj,cn1,cn2, tar_ei, filled1, [])
    

class CNLinkPredictorOringin(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        
        
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_buffer("innerprod", torch.tensor([0.0]))
        self.n = 0

    def innerprod1(self, E1, E2):
        if self.training:
            hprod = spsphadamard(E1, E2)
            innerprod = hprod.values.sum()
            self.n += 1
            beta = self.n**-1
            self.innerprod *= (1 - beta)
            self.innerprod += beta * innerprod
        # print("innerprod", self.innerprod)
        return self.innerprod

    def multidomainforward(self,
                           x,
                           adj,
                           cn1,
                           cn2,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        
        col_sum = cn1.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        inv_col_sum[~non_empty_cols] = 0  
        

        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn1 = cn1.mul(inv_col_sum)  
        
        num_rows = normalized_cn1.size(0) 
        
        num_cols = normalized_cn1.size(1)  

        Ortho_zero = SparseTensor(
            row=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            col=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            value=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            sparse_sizes=(num_rows, num_cols)
        )



        spnormalized_cn1=normalized_cn1.to_torch_sparse_coo_tensor()
        spOrtho_zero=Ortho_zero.to_torch_sparse_coo_tensor()
        




       

        spcn2=cn2.to_torch_sparse_coo_tensor()



        spcn2_copy=spcn2
        spnormalized_cn1_copy=spnormalized_cn1        
        spcn2 = spcn2.coalesce()
        spnormalized_cn1 = spnormalized_cn1.coalesce()


        num_nodes = spnormalized_cn1.shape[1] 
        num_edges = spnormalized_cn1.shape[0]

        row1, col1 = spnormalized_cn1.indices()
        row2, col2 = spcn2.indices()

 
        value1 = spnormalized_cn1.values()
        value2 = spcn2.values()



        spnormalized_cn1 = torch_sparse.SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=(num_edges, num_nodes))
        spcn2 = torch_sparse.SparseTensor(row=row2, col=col2, value=value2, sparse_sizes=(num_edges, num_nodes))



        row, col, val = spcn2.coo()
        spcn2 = pSparseTensor(torch.stack((row, col)), val, spcn2.sizes(), is_coalesced=True)

        row, col, val = spnormalized_cn1.coo()
        spnormalized_cn1 = pSparseTensor(torch.stack((row, col)), val, spnormalized_cn1.sizes(), is_coalesced=True)


        inner_product = self.innerprod1(spcn2, spnormalized_cn1)

           



        spcn2 = spcn2_copy.coalesce()
        spnormalized_cn1 = spnormalized_cn1_copy.coalesce()
        spOrtho_zero = spOrtho_zero.coalesce()

           
           
            
        spcn2_indices = spcn2.indices()
        spnormalized_cn1_indices = spnormalized_cn1.indices()
        spOrtho_zero_indices = spOrtho_zero.indices()

        spcn2_values = spcn2.values()
        spnormalized_cn1_values = spnormalized_cn1.values()
        spOrtho_zero_values = spOrtho_zero.values()

            
        unique_indices, inverse_indices = torch.unique(
                torch.cat((spcn2_indices, spnormalized_cn1_indices, spOrtho_zero_indices), dim=1),
                dim=1,
                return_inverse=True
            )

        spcn2_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spcn2_aligned_values[inverse_indices[:spcn2_indices.size(1)]] = spcn2_values

        spnormalized_cn1_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spnormalized_cn1_aligned_values[inverse_indices[spcn2_indices.size(1):spcn2_indices.size(1) + spnormalized_cn1_indices.size(1)]] = spnormalized_cn1_values

        spOrtho_zero_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        if spOrtho_zero._nnz() > 0:
            spOrtho_zero_aligned_values[inverse_indices[spcn2_indices.size(1) + spnormalized_cn1_indices.size(1):]] = spOrtho_zero_values

            
            
        if spnormalized_cn1_aligned_values.numel() > 0:
                
            scale_factor = spnormalized_cn1_aligned_values.abs().max().item()
        else:
        
            scale_factor = 1.0  
        normalized_inner_product = inner_product / scale_factor if scale_factor > 0 else inner_product



        new_values = (
                spcn2_aligned_values
                - normalized_inner_product * spnormalized_cn1_aligned_values
                
            )
            
        spcn2 = torch.sparse_coo_tensor(
                unique_indices,
                new_values,
                size=spcn2.size(),
                device=spcn2.device
            ).coalesce()





            
        spcn2 = spcn2.coalesce()

            
        indices = spcn2.indices()
        values = spcn2.values()

            
        col_sum = torch.zeros(spcn2.size(1), device=spcn2.device)
        col_sum.index_add_(0, indices[1], values)  

            
        col_sum[col_sum == 0] = 1
        inv_col_sum = 1 / col_sum

            
        normalized_values = values * inv_col_sum[indices[1]]

            
        row, col = indices
        normalized_spcn2_sparse = SparseTensor(
                row=row,
                col=col,
                value=normalized_values,
                sparse_sizes=spcn2.size(),
                is_sorted=True  
            )

           
        xcn1=spmm_add(normalized_cn1, x)
        xcn2=spmm_add(normalized_spcn2_sparse, x)

        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        
        
        xcn1 = self.xcn1lin(xcn1)
        
        xcn2 = self.xcn2lin(xcn2)

        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 + self.beta * xij)
        
        
        return x
    
    def forward(self, x, adj,cn1,cn2, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj,cn1,cn2, tar_ei, filled1, [])



class CNLinkPredictor3hopCNs(nn.Module):
    cndeg: Final[int]

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta * torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(
                inplace=True), nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
            if not tailact else nn.Identity())
        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn3lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))

        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
            if not tailact else nn.Identity())
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
            if twolayerlin else nn.Identity(),
            lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
            nn.Dropout(dropout, inplace=True)
            if twolayerlin else nn.Identity(),
            nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
            nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_buffer("innerprod", torch.tensor([0.0]))
        self.n = 0

    def innerprod1(self, E1, E2):
        if self.training:
            hprod = spsphadamard(E1, E2)
            innerprod = hprod.values.sum()
            self.n += 1
            beta = self.n**-1
            self.innerprod *= (1 - beta)
            self.innerprod += beta * innerprod
        # print("innerprod", self.innerprod)
        return self.innerprod

    def multidomainforward(self,
                           x,
                           adj: pSparseTensor,
                           cn1,
                           cn2,
                           cn3,
                           tar_ei,
                           args,
                           cndropprobs: Iterable[float] = []):


        col_sum = cn1.sum(dim=0)
            
        col_sum[col_sum == 0] = 1
 

            
        inv_col_sum = 1 / col_sum
            
        non_empty_cols = col_sum != 1
           
        inv_col_sum[~non_empty_cols] = 0
            
        inv_col_sum = inv_col_sum.view(1, -1)
           
        normalized_cn1 = cn1.mul(inv_col_sum)
            

        num_rows = normalized_cn1.size(0)
        num_cols = normalized_cn1.size(1)
            

        Ortho_zero = SparseTensor(
                row=torch.tensor([], dtype=torch.long, device=normalized_cn1.device()),
                col=torch.tensor([], dtype=torch.long, device=normalized_cn1.device()),
                value=torch.tensor([], dtype=torch.long, device=normalized_cn1.device()),
                sparse_sizes=(num_rows, num_cols)
            )
    

        spnormalized_cn1 = normalized_cn1.to_torch_sparse_coo_tensor()
 

        spOrtho_zero = Ortho_zero.to_torch_sparse_coo_tensor()
        

        spcn2 = cn2.to_torch_sparse_coo_tensor()
            
        spcn2_copy = spcn2
            
        spnormalized_cn1_copy = spnormalized_cn1
            
        spcn2 = spcn2.coalesce()
            

        spnormalized_cn1 = spnormalized_cn1.coalesce()
            
        num_nodes = spnormalized_cn1.shape[1]
        num_edges = spnormalized_cn1.shape[0]
            
        row1, col1 = spnormalized_cn1.indices()
            
        row2, col2 = spcn2.indices()
           
        value1 = spnormalized_cn1.values()
        value2 = spcn2.values()
           
        spnormalized_cn1 = torch_sparse.SparseTensor(
                row=row1, 
                col=col1, 
                value=value1, 
                sparse_sizes=(num_edges, num_nodes)
            )
            
        spcn2 = torch_sparse.SparseTensor(
                row=row2, 
                col=col2, 
                value=value2, 
                sparse_sizes=(num_edges, num_nodes)
            )
            
        row, col, val = spcn2.coo()
           
        spcn2 = pSparseTensor(
                torch.stack((row, col)), 
                val, 
                spcn2.sizes(), 
                is_coalesced=True
            )
            

        row, col, val = spnormalized_cn1.coo()
            

        spnormalized_cn1 = pSparseTensor(
                torch.stack((row, col)), 
                val, 
                spnormalized_cn1.sizes(), 
                is_coalesced=True
            )
            
        inner_product = self.innerprod1(spcn2, spnormalized_cn1)
            
        spcn2 = spcn2_copy.coalesce()
           

        spnormalized_cn1 = spnormalized_cn1_copy.coalesce()
            

        spOrtho_zero = spOrtho_zero.coalesce()
           
        spcn2_indices = spcn2.indices()
        spnormalized_cn1_indices = spnormalized_cn1.indices()
        spOrtho_zero_indices = spOrtho_zero.indices()
        spcn2_values = spcn2.values()
        spnormalized_cn1_values = spnormalized_cn1.values()
        spOrtho_zero_values = spOrtho_zero.values()

           
        combined_indices = torch.cat((spcn2_indices, spnormalized_cn1_indices, spOrtho_zero_indices), dim=1)
        unique_indices, inverse_indices = torch.unique(
                combined_indices,
                dim=1,
                return_inverse=True
        )
            
        spcn2_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spcn2_aligned_values[inverse_indices[:spcn2_indices.size(1)]] = spcn2_values
           
        start_idx_cn1 = spcn2_indices.size(1)
        end_idx_cn1 = start_idx_cn1 + spnormalized_cn1_indices.size(1)
        spnormalized_cn1_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        spnormalized_cn1_aligned_values[inverse_indices[start_idx_cn1:end_idx_cn1]] = spnormalized_cn1_values
           
        start_idx_ortho = end_idx_cn1
        spOrtho_zero_aligned_values = torch.zeros(unique_indices.size(1), device=spcn2.device)
        if spOrtho_zero._nnz() > 0:
            spOrtho_zero_aligned_values[inverse_indices[start_idx_ortho:]] = spOrtho_zero_values
            
        if spnormalized_cn1_aligned_values.numel() > 0:
            scale_factor = spnormalized_cn1_aligned_values.abs().max().item()
                
        else:
            scale_factor = 1.0
                
        if scale_factor > 0:
            normalized_inner_product = inner_product / scale_factor
                
        else:
            normalized_inner_product = inner_product
                
        new_values = (
                spcn2_aligned_values
                - normalized_inner_product * spnormalized_cn1_aligned_values   
            )
            
        spcn2 = torch.sparse_coo_tensor(
                unique_indices,
                new_values,
                size=spcn2.size(),
                device=spcn2.device
            ).coalesce()
           
        spcn2 = spcn2.coalesce()
            
        indices = spcn2.indices()
        values = spcn2.values()
           
        col_sum = torch.zeros(spcn2.size(1), device=spcn2.device)
        col_sum.index_add_(0, indices[1], values)
           
        col_sum_zero_before = torch.sum(col_sum == 0).item()
        col_sum[col_sum == 0] = 1
            
        inv_col_sum = 1 / col_sum
           
        normalized_values = values * inv_col_sum[indices[1]]
            
        row, col = indices
           
        normalized_spcn2_sparse = SparseTensor(
                row=row,
                col=col,
                value=normalized_values,
                sparse_sizes=spcn2.size(),
                is_sorted=True  
            )
           
        xcn1 = spmm_add(normalized_cn1, x)


        xcn2 = spmm_add(normalized_spcn2_sparse, x)





        '''col_sum = cn3.sum(dim=0)
        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        inv_col_sum[~non_empty_cols] = 0  
        inv_col_sum = inv_col_sum.view(1, -1) 
        normalized_cn3 = cn3.mul(inv_col_sum) 
        spnormalized_cn3=normalized_cn3.to_torch_sparse_coo_tensor()'''


        normalized_cn1=normalized_cn1
        normalized_spcn2_sparse=normalized_spcn2_sparse


        spnormalized_cn1 = normalized_cn1.to_torch_sparse_coo_tensor()
            
        normalized_spcn2_sparse = normalized_spcn2_sparse.to_torch_sparse_coo_tensor()
            
        spcn3 = cn3.to_torch_sparse_coo_tensor()
            
        spnormalized_cn1_copy = spnormalized_cn1
            

        normalized_spcn2_sparse_copy = normalized_spcn2_sparse
            

        spcn3_copy = spcn3
            

        spcn3 = spcn3.coalesce()
            

        spnormalized_cn1 = spnormalized_cn1.coalesce()
            

        normalized_spcn2_sparse = normalized_spcn2_sparse.coalesce()
            
        num_nodes = spnormalized_cn1.shape[1]
        num_edges = spnormalized_cn1.shape[0]
            
        row1, col1 = spnormalized_cn1.indices()
            

        value1 = spnormalized_cn1.values()
            
        row2, col2 = spcn3.indices()
            

        value2 = spcn3.values()
           
        spnormalized_cn1 = torch_sparse.SparseTensor(
                row=row1,
                col=col1,
                value=value1,
                sparse_sizes=(num_edges, num_nodes)
            )
            

        spcn3 = torch_sparse.SparseTensor(
                row=row2,
                col=col2,
                value=value2,
                sparse_sizes=(num_edges, num_nodes)
            )
            
        row, col, val = spcn3.coo()
            
        spcn3 = pSparseTensor(
                torch.stack((row, col)),
                val,
                spcn3.sizes(),
                is_coalesced=True
            )
            

        row, col, val = spnormalized_cn1.coo()
            
        spnormalized_cn1 = pSparseTensor(
                torch.stack((row, col)),
                val,
                spnormalized_cn1.sizes(),
                is_coalesced=True
            )
            

        inner_product1 = self.innerprod1(spcn3, spnormalized_cn1)
           
        num_nodes = normalized_spcn2_sparse.shape[1]
        num_edges = normalized_spcn2_sparse.shape[0]
           
        row3, col3 = normalized_spcn2_sparse.indices()
            
        value3 = normalized_spcn2_sparse.values()
            
        normalized_spcn2_sparse = torch_sparse.SparseTensor(
                row=row3, 
                col=col3, 
                value=value3, 
                sparse_sizes=(num_edges, num_nodes)
            )
            
        row, col, val = normalized_spcn2_sparse.coo()
            
        normalized_spcn2_sparse = pSparseTensor(
                torch.stack((row, col)), 
                val, 
                normalized_spcn2_sparse.sizes(), 
                is_coalesced=True
            )
           
        inner_product2 = self.innerprod1(spcn3, normalized_spcn2_sparse)
           
        spcn3 = spcn3_copy.coalesce()
            

        spnormalized_cn1 = spnormalized_cn1_copy.coalesce()
            

        normalized_spcn2_sparse = normalized_spcn2_sparse_copy.coalesce()
            
        spcn3_indices = spcn3.indices()
        spnormalized_cn1_indices = spnormalized_cn1.indices()
        normalized_spcn2_sparse_indices = normalized_spcn2_sparse.indices()

        spcn3_values = spcn3.values()
        spnormalized_cn1_values = spnormalized_cn1.values()
        normalized_spcn2_sparse_values = normalized_spcn2_sparse.values()

         
        combined_indices = torch.cat((spcn3_indices, spnormalized_cn1_indices, normalized_spcn2_sparse_indices), dim=1)
        unique_indices, inverse_indices = torch.unique(
                combined_indices,
                dim=1,
                return_inverse=True
            )
           
        spcn3_aligned_values = torch.zeros(unique_indices.size(1), device=spcn3.device)
        spcn3_aligned_values[inverse_indices[:spcn3_indices.size(1)]] = spcn3_values
            
        start_idx_cn1 = spcn3_indices.size(1)
        end_idx_cn1 = start_idx_cn1 + spnormalized_cn1_indices.size(1)
        spnormalized_cn1_aligned_values = torch.zeros(unique_indices.size(1), device=spcn3.device)
        spnormalized_cn1_aligned_values[inverse_indices[start_idx_cn1:end_idx_cn1]] = spnormalized_cn1_values
            
        start_idx_spcn2 = end_idx_cn1
        spOrtho_zero_aligned_values = torch.zeros(unique_indices.size(1), device=spcn3.device)
        normalized_spcn2_sparse_aligned_values = torch.zeros(unique_indices.size(1), device=spcn3.device)
        normalized_spcn2_sparse_aligned_values[inverse_indices[start_idx_spcn2:]] = normalized_spcn2_sparse_values
          
        if spnormalized_cn1_aligned_values.numel() > 0:
            scale_factor = spnormalized_cn1_aligned_values.abs().max().item()
                
        else:
            scale_factor = 1.0
               
        if scale_factor > 0:
            normalized_inner_product1 = inner_product1 / scale_factor
            normalized_inner_product2 = inner_product2 / scale_factor
             
        else:
            normalized_inner_product1 = inner_product1
            normalized_inner_product2 = inner_product2
               
        new_values = (
                spcn3_aligned_values
                - normalized_inner_product1 * spnormalized_cn1_aligned_values
                - normalized_inner_product2 * normalized_spcn2_sparse_aligned_values   
            )
            
        spcn3 = torch.sparse_coo_tensor(
                unique_indices,
                new_values,
                size=spcn3.size(),
                device=spcn3.device
            ).coalesce()
           
        spcn3 = spcn3.coalesce()
         
        indices = spcn3.indices()
        values = spcn3.values()
    
        col_sum = torch.zeros(spcn3.size(1), device=spcn3.device)
            

        col_sum.index_add_(0, indices[1], values)
   
        col_sum_zero_before = torch.sum(col_sum == 0).item()
        col_sum[col_sum == 0] = 1
     
        inv_col_sum = 1 / col_sum
           
        normalized_values = values * inv_col_sum[indices[1]]
           
        row, col = indices
           
        normalized_spcn3_sparse = SparseTensor(
                row=row,
                col=col,
                value=normalized_values,
                sparse_sizes=spcn3.size(),
                is_sorted=True  
            )
            
        xcn3 = spmm_add(normalized_spcn3_sparse, x)
            



        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        
        xcn1 = self.xcn1lin(xcn1)
        
        xcn2 = self.xcn2lin(xcn2)
        xcn3 = self.xcn3lin(xcn3)
        
        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 + alpha[2] * xcn3+self.beta * xij)
        
        
        return x
    
    def forward(self, x, adj, cn1,cn2,cn3,tar_ei,args):
        return self.multidomainforward(x, adj, cn1,cn2,cn3,tar_ei,args)



def T0(x):
    return torch.ones_like(x)

def T1(x):
    return x

def T2(x):
    return 2 * x**2 - 1

def T3(x):
    return 4 * x**3 - 3 * x

def T4(x):
    return 8 * x**4 - 8 * x**2 + 1

def T5(x):
    return 16 * x**5 - 20 * x**3 + 5 * x

def T6(x):
    return 32 * x**6 - 48 * x**4 + 18 * x**2 - 1

def T7(x):
    return 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x

def T8(x):
    return 128 * x**8 - 256 * x**6 + 160 * x**4 - 32 * x**2 + 1

def T9(x):
    return 256 * x**9 - 576 * x**7 + 432 * x**5 - 120 * x**3 + 9 * x

def T10(x):
    return 512 * x**10 - 1280 * x**8 + 1120 * x**6 - 400 * x**4 + 50 * x**2 - 1


polynomials = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]


def evaluate_polynomial(normalized_cn1_size_1, poly_index):
    
    if poly_index < 0 or poly_index >= len(polynomials):
        raise ValueError(f"Invalid poly_index. Must be between 0 and {len(polynomials)-1}.")
    
    
    x_values = torch.linspace(-1, 1, normalized_cn1_size_1)
    
    
    selected_polynomial = polynomials[poly_index]
    
    
    y_values = selected_polynomial(x_values).unsqueeze(1)
    
    
    indices = torch.arange(normalized_cn1_size_1)  
    rows = indices
    cols = indices
    
    
    values = y_values.squeeze(1)  
    sparse_diag_matrix = SparseTensor(row=rows, col=cols, value=values, sparse_sizes=(normalized_cn1_size_1, normalized_cn1_size_1))
    
    
    return sparse_diag_matrix

class CNLinkPredictorbaselearn(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        
        
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_buffer("innerprod", torch.tensor([0.0]))
        self.n = 0

    def innerprod1(self, E1, E2):
        if self.training:
            hprod = spsphadamard(E1, E2)
            innerprod = hprod.values.sum()
            self.n += 1
            beta = self.n**-1
            self.innerprod *= (1 - beta)
            self.innerprod += beta * innerprod
        # print("innerprod", self.innerprod)
        return self.innerprod

    def multidomainforward(self,
                           x,
                           adj,
                           cn1,
                           cn2,
                           tar_ei,
                           args,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []
                           ):
        
        
        col_sum = cn1.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        
        inv_col_sum[~non_empty_cols] = args.sum  


        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn1 = cn1.mul(inv_col_sum)  
        
        num_rows = normalized_cn1.size(0) 
        
        num_cols = normalized_cn1.size(1)  

        Ortho_zero = SparseTensor(
            row=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            col=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            value=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            sparse_sizes=(num_rows, num_cols)
        )

        devv=normalized_cn1.device()
        
        base1=evaluate_polynomial(num_cols,0).to(normalized_cn1.device())

        row, col, val = normalized_cn1.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        normalized_cn1 = pSparseTensor(torch.stack((row, col)), val, normalized_cn1.sizes(), is_coalesced=True)

        row, col, val = base1.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        base1 = pSparseTensor(torch.stack((row, col)), val, base1.sizes(), is_coalesced=True)  
        
        cn1_res=spspmm(normalized_cn1, 1, base1, 0)
        cn1_res=cn1_res.to_torch_sparse_coo()

        rows,cols = cn1_res.indices()
        values = cn1_res.values()
        
        cn1 = SparseTensor(
                row=rows,
                col=cols,
                value=values,
                sparse_sizes=cn1_res.size(),
                is_sorted=True  
            )
        

        col_sum = cn2.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        
        inv_col_sum[~non_empty_cols] = args.sum  


        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn2 = cn2.mul(inv_col_sum)
    
        
        base2=evaluate_polynomial(num_cols,0).to(devv)

        
        row, col, val = cn2.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        spcn2 = pSparseTensor(torch.stack((row, col)), val, cn2.sizes(), is_coalesced=True)

        row, col, val = base2.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        base2 = pSparseTensor(torch.stack((row, col)), val, base2.sizes(), is_coalesced=True)  
        
        cn2_res=spspmm(spcn2, 1, base2, 0)
        
        cn2_res=cn2_res.to_torch_sparse_coo()

        rows,cols = cn2_res.indices()
        values = cn2_res.values()
        
        cn2 = SparseTensor(
                row=rows,
                col=cols,
                value=values,
                sparse_sizes=cn2_res.size(),
                is_sorted=True  
            )
        
        
           
        xcn1=spmm_add(cn1, x)
        xcn2=spmm_add(cn2, x)

        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        
        xcn1 = self.xcn1lin(xcn1)
        
        xcn2 = self.xcn2lin(xcn2)

        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 + self.beta * xij)
        
        
        return x
    
    def forward(self, x, adj,cn1,cn2, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj,cn1,cn2, tar_ei, filled1, [])



class CNLinkPredictorbaselearnablation(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        
        
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(3*hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_buffer("innerprod", torch.tensor([0.0]))
        self.n = 0

    def innerprod1(self, E1, E2):
        if self.training:
            hprod = spsphadamard(E1, E2)
            innerprod = hprod.values.sum()
            self.n += 1
            beta = self.n**-1
            self.innerprod *= (1 - beta)
            self.innerprod += beta * innerprod
        # print("innerprod", self.innerprod)
        return self.innerprod

    def multidomainforward(self,
                           x,
                           adj,
                           cn1,
                           cn2,
                           tar_ei,
                           args,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []
                           ):
        
        
        col_sum = cn1.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        
        inv_col_sum[~non_empty_cols] = args.sum  


        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn1 = cn1.mul(inv_col_sum)  

        normalized_cn1 = cn1
        
        num_rows = normalized_cn1.size(0) 
        
        num_cols = normalized_cn1.size(1)  

        Ortho_zero = SparseTensor(
            row=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            col=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            value=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            sparse_sizes=(num_rows, num_cols)
        )

        devv=normalized_cn1.device()
        
        base1=evaluate_polynomial(num_cols,0).to(normalized_cn1.device())

        row, col, val = normalized_cn1.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        normalized_cn1 = pSparseTensor(torch.stack((row, col)), val, normalized_cn1.sizes(), is_coalesced=True)

        row, col, val = base1.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        base1 = pSparseTensor(torch.stack((row, col)), val, base1.sizes(), is_coalesced=True)  
        
        cn1_res=spspmm(normalized_cn1, 1, base1, 0)
        cn1_res=cn1_res.to_torch_sparse_coo()

        rows,cols = cn1_res.indices()
        values = cn1_res.values()
        
        cn1 = SparseTensor(
                row=rows,
                col=cols,
                value=values,
                sparse_sizes=cn1_res.size(),
                is_sorted=True  
            )
        

        col_sum = cn2.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        
        inv_col_sum[~non_empty_cols] = args.sum 


        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn2 = cn2.mul(inv_col_sum)
        normalized_cn2 = cn2
    
        base2=evaluate_polynomial(num_cols,0).to(devv)

        
        row, col, val = cn2.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        spcn2 = pSparseTensor(torch.stack((row, col)), val, cn2.sizes(), is_coalesced=True)

        row, col, val = base2.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        base2 = pSparseTensor(torch.stack((row, col)), val, base2.sizes(), is_coalesced=True)  
        
        cn2_res=spspmm(spcn2, 1, base2, 0)
        
        cn2_res=cn2_res.to_torch_sparse_coo()

        rows,cols = cn2_res.indices()
        values = cn2_res.values()
        
        cn2 = SparseTensor(
                row=rows,
                col=cols,
                value=values,
                sparse_sizes=cn2_res.size(),
                is_sorted=True  
            )
        
        
           
        xcn1=spmm_add(cn1, x)
        xcn2=spmm_add(cn2, x)

        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        
        
        xcn1 = self.xcn1lin(xcn1)
        
        xcn2 = self.xcn2lin(xcn2)

        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 + self.beta * xij)
        #x = self.lin(torch.cat([alpha[0] * xcn1, alpha[1] * xcn2, self.beta * xij], dim=1))
        
        return x
    
    def forward(self, x, adj,cn1,cn2, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj,cn1,cn2, tar_ei, filled1, [])





class CNLinkPredictorbaselearnablationwithoutx(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        
        
        self.xijlin = nn.Sequential(
            nn.Linear(32, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_buffer("innerprod", torch.tensor([0.0]))
        self.n = 0

    def innerprod1(self, E1, E2):
        if self.training:
            hprod = spsphadamard(E1, E2)
            innerprod = hprod.values.sum()
            self.n += 1
            beta = self.n**-1
            self.innerprod *= (1 - beta)
            self.innerprod += beta * innerprod
        # print("innerprod", self.innerprod)
        return self.innerprod

    def multidomainforward(self,
                           x,
                           adj,
                           cn1,
                           cn2,
                           tar_ei,
                           args,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []
                           ):
        
        
        col_sum = cn1.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        
        inv_col_sum[~non_empty_cols] = args.sum  


        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn1 = cn1.mul(inv_col_sum)  
        
        num_rows = normalized_cn1.size(0) 
        
        num_cols = normalized_cn1.size(1)  

        Ortho_zero = SparseTensor(
            row=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            col=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            value=torch.tensor([], dtype=torch.long,device=normalized_cn1.device()),
            sparse_sizes=(num_rows, num_cols)
        )

        devv=normalized_cn1.device()
        
        base1=evaluate_polynomial(num_cols,0).to(normalized_cn1.device())
        
        row, col, val = normalized_cn1.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        normalized_cn1 = pSparseTensor(torch.stack((row, col)), val, normalized_cn1.sizes(), is_coalesced=True)

        row, col, val = base1.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        base1 = pSparseTensor(torch.stack((row, col)), val, base1.sizes(), is_coalesced=True)  
       
        cn1_res=spspmm(normalized_cn1, 1, base1, 0)
        cn1_res=cn1_res.to_torch_sparse_coo()

        
        rows,cols = cn1_res.indices()
        values = cn1_res.values()
        
        normalized_cn1 = SparseTensor(
                row=rows,
                col=cols,
                value=values,
                sparse_sizes=cn1_res.size(),
                is_sorted=True  
            )

        col_sum = cn2.sum(dim=0)

        col_sum[col_sum == 0] = 1  
        inv_col_sum = 1 / col_sum
        non_empty_cols = col_sum != 1  
        
        inv_col_sum[~non_empty_cols] = args.sum  


        inv_col_sum = inv_col_sum.view(1, -1)  

        
        normalized_cn2 = cn2.mul(inv_col_sum)
    
        base2=evaluate_polynomial(num_cols,0).to(devv)

        

        
        row, col, val = cn2.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        spcn2 = pSparseTensor(torch.stack((row, col)), val, cn2.sizes(), is_coalesced=True)

        row, col, val = base2.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        base2 = pSparseTensor(torch.stack((row, col)), val, base2.sizes(), is_coalesced=True)  
       
        cn2_res=spspmm(spcn2, 1, base2, 0)
        
        cn2_res=cn2_res.to_torch_sparse_coo()

        rows,cols = cn2_res.indices()
        values = cn2_res.values()
        
        normalized_spcn2_sparse = SparseTensor(
                row=rows,
                col=cols,
                value=values,
                sparse_sizes=cn2_res.size(),
                is_sorted=True  
            )




        n = normalized_cn1.size(1) 
        dev = normalized_cn1.device() 

       
        indices = torch.arange(n, device=normalized_cn1.device()).unsqueeze(0)
        indices = torch.cat([indices, indices], dim=0)  
        values = torch.ones(n, device=normalized_cn1.device())  

       
        sparse_identity_matrix = torch.sparse_coo_tensor(indices, values, (n, n), device=normalized_cn1.device())

        sparse_identity_matrix = sparse_identity_matrix.coalesce()

        num_nodes = sparse_identity_matrix.shape[1] 
        num_edges = sparse_identity_matrix.shape[0]

        row1, col1 = sparse_identity_matrix.indices()
        

 
        value1 = sparse_identity_matrix.values()
        

        sparse_identity_matrix = torch_sparse.SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=(num_edges, num_nodes))

        row, col, val = sparse_identity_matrix.coo()
        sparse_identity_matrix = pSparseTensor(torch.stack((row, col)), val, sparse_identity_matrix.sizes(), is_coalesced=True)



        row, col, val = normalized_cn1.coo()
        normalized_cn1 = pSparseTensor(torch.stack((row, col)), val, normalized_cn1.sizes(), is_coalesced=True)

        row, col, val = normalized_spcn2_sparse.coo()
        normalized_spcn2_sparse = pSparseTensor(torch.stack((row, col)), val, normalized_spcn2_sparse.sizes(), is_coalesced=True)


        xcn1=spspmm(normalized_cn1, 1, sparse_identity_matrix, 0)
        xcn2=spspmm(normalized_spcn2_sparse, 1, sparse_identity_matrix, 0)




        xcn1=xcn1.to_torch_sparse_coo()
        xcn2=xcn2.to_torch_sparse_coo()
        xcn1=xcn1.coalesce()
        xcn2=xcn2.coalesce()
        num_edge=xcn1.size(0)
        num_node=xcn1.size(1)
        indices1 = xcn1.indices()
        indices2= xcn2.indices()
        values1 = xcn1.values()
        values2 = xcn2.values()
        xcn1 = torch.sparse_coo_tensor(indices1, values1, (num_edge, num_node), device=dev)
        xcn2 = torch.sparse_coo_tensor(indices2, values2, (num_edge, num_node), device=dev)


        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        xcn1 = self.xcn1lin(xcn1)
        
        xcn2 = self.xcn2lin(xcn2)



        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 + self.beta * xij)
        
        
        return x
    
    def forward(self, x, adj,cn1,cn2, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj,cn1,cn2, tar_ei, filled1, [])


predictor_dict = {

    "cn1": CNLinkPredictor,
    "cn2": IncompleteCN1Predictor,
    "cn3": IncompleteCN1Predictorhighorder,
    "cn4": IncompleteCN1PredictorSaveMemory,
    "cn5": CNLinkPredictorOringin,
    "cn6": CNLinkPredictor3hopCNs,
    "cn7": CNLinkPredictorbaselearn,
    "cn8": CNLinkPredictorbaselearnablation,
    "cn9": CNLinkPredictorbaselearnablationwithoutx

}