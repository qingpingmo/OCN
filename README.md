This repository contains the official code for the paper:

OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction

**Environment**



```
conda env create -f environment.yml
```



**Prepare Datasets**

```
python ogbdataset.py
```

**Reproduce Results**


### OCN
Cora
```
python NeighborOverlap_large.py --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor cn5 --dataset Cora  --epochs 100 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --sum 0 
```

Citeseer
```
python NeighborOverlap_large.py --dataset Citeseer --predictor cn5 --epochs 100 --runs 10 --batch_size 384 --testbs 4096 --use_xlin --maskinput --lnnn --res --jk --model puremean --mplayers 3 --nnlayers 1 --hiddim 64 --gnndp 0.12 --xdp 0.73 --tdp 0.88 --gnnedp 0.07 --predp 0.19 --preedp 0.66 --gnnlr 0.0009 --prelr 0.00096 --beta 4.36 --alpha 2.48 --probscale 6.19 --proboffset 9.69 --pt 0.042 --cnprob 0.94 --sum 27.29
```

Pubmed
```
python NeighborOverlap_large.py   --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1  --probscale 5.3 --proboffset 0.5 --alpha 0.3  --gnnlr 0.0097 --prelr 0.002  --batch_size 2048  --ln --lnnn --predictor cn5 --dataset Pubmed  --epochs 200 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --sum 1
```

collab
```
python NeighborOverlap_large.py --use_valedges_as_input --epochs 200 --runs 10 --dataset collab --batch_size 65536 --testbs 65536 --maskinput --mplayers 1 --nnlayers 3 --hiddim 256 --ln --lnnn --jk --gnndp 0.05 --xdp 0.7 --tdp 0.3 --gnnedp 0.0 --predp 0.05 --preedp 0.4 --gnnlr 0.0043 --prelr 0.0024 --beta 1 --alpha 1.0 --use_xlin --tailact --probscale 4.3 --proboffset 2.8 --pt 0.75 --predictor cn5 --depth 1 --model gin
```

ppa
```
python NeighborOverlap_large_ppa.py   --sum 0 --epochs 20 --runs 10 --dataset ppa --batch_size 16384 --linkbatchsize 2048 --testbs 2048 --maskinput --mplayers 1 --nnlayers 3 --hiddim 64 --ln --lnnn --jk --gnndp 0.0 --xdp 0.0 --tdp 0.0 --gnnedp 0.0 --predp 0.0 --preedp 0.0 --gnnlr 0.0013 --prelr 0.0013 --beta 1 --alpha 1.0 --use_xlin  --tailact --probscale 4.3 --proboffset 2.8 --pt 0.75 --predictor cn5 --depth 1 --model gcn --cnprob 0
```


citation2
```
python NeighborOverlapCitation2.py --dataset citation2 --predictor cn5 --epochs 20 --runs 10 --batch_size 16384 --testbs 2048 --linkbatchsize 2048 --use_xlin --ln --res --jk --tailact --model gcn --mplayers 5 --nnlayers 3 --hiddim 32 --depth 3 --gnndp 0.28 --xdp 0.5 --tdp 0.28 --gnnedp 0.20 --predp 0.10 --preedp 0.12 --gnnlr 0.00023 --prelr 0.0008 --beta 0.23 --alpha 1.33 --probscale 2.64 --proboffset 4.5 --pt 0.34 --cnprob 0.78 --sum 1
```


ddi
```
python NeighborOverlap_large.py --dataset ddi --predictor cn5 --epochs 100 --runs 10 --batch_size 32768 --testbs 32768 --use_xlin --adj2byblock --maskinput --lnnn --res --learnpt --model puregcn --mplayers 3 --nnlayers 3 --hiddim 64 --gnndp 0.25 --xdp 0.13 --tdp 0.38 --gnnedp 0.51 --predp 0.10 --preedp 0.13 --gnnlr 0.0009 --prelr 0.00083 --beta 0.33 --alpha 7.18--probscale 4.31 --proboffset 4.11 --pt 0.73 --cnprob 0.93 --sum 2.74
```


### OCNP
Cora
```
python NeighborOverlap_large.py --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1024  --ln --lnnn --predictor cn7 --dataset Cora  --epochs 200 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --sum 0
```

Citeseer
```
python NeighborOverlap_large.py --dataset Citeseer --predictor cn7 --epochs 100 --runs 10 --batch_size 384 --testbs 4096 --use_xlin --maskinput --lnnn --res --jk --model puremean --mplayers 3 --nnlayers 1 --hiddim 64 --gnndp 0.12 --xdp 0.73 --tdp 0.88 --gnnedp 0.07 --predp 0.19 --preedp 0.66 --gnnlr 0.0009 --prelr 0.00096 --beta 4.36 --alpha 2.48 --probscale 6.19 --proboffset 9.69 --pt 0.042 --cnprob 0.94 --sum 27.29
```

Pubmed
```
python NeighborOverlap_large.py   --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 --gnndp 0.1  --probscale 5.3 --proboffset 0.5 --alpha 0.3  --gnnlr 0.0097 --prelr 0.002  --batch_size 2048  --ln --lnnn --predictor cn7 --dataset Pubmed  --epochs 200 --runs 10 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --sum 1
```

collab
```
python NeighborOverlap_large.py    --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor cn7 --dataset collab  --epochs 100 --runs 10 --model gin --hiddim 256 --mplayers 1  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact
```

ppa
```
python NeighborOverlap_large_ppa.py   --sum 0 --epochs 20 --runs 10 --dataset ppa --batch_size 16384 --linkbatchsize 2048 --testbs 2048 --maskinput --mplayers 1 --nnlayers 3 --hiddim 64 --ln --lnnn --jk --gnndp 0.0 --xdp 0.0 --tdp 0.0 --gnnedp 0.0 --predp 0.0 --preedp 0.0 --gnnlr 0.0013 --prelr 0.0013 --beta 1 --alpha 1.0 --use_xlin  --tailact --probscale 4.3 --proboffset 2.8 --pt 0.75 --predictor cn7 --depth 1 --model gcn --cnprob 0
```


citation2
```
python NeighborOverlapCitation2.py --dataset citation2 --predictor cn7 --epochs 20 --runs 10 --batch_size 16384 --testbs 2048 --linkbatchsize 2048 --use_xlin --ln --res --jk --tailact --model gcn --mplayers 5 --nnlayers 3 --hiddim 32 --depth 3 --gnndp 0.28 --xdp 0.5 --tdp 0.28 --gnnedp 0.20 --predp 0.10 --preedp 0.12 --gnnlr 0.00023 --prelr 0.0008 --beta 0.23 --alpha 1.33 --probscale 2.64 --proboffset 4.5 --pt 0.34 --cnprob 0.78 --sum 1
```


ddi
```
python NeighborOverlap_large.py --dataset ddi --predictor cn7 --epochs 100 --runs 10 --batch_size 32768 --testbs 32768 --use_xlin --adj2byblock --maskinput --lnnn --res --learnpt --model puregcn --mplayers 3 --nnlayers 3 --hiddim 64 --gnndp 0.25 --xdp 0.13 --tdp 0.38 --gnnedp 0.51 --predp 0.10 --preedp 0.13 --gnnlr 0.0009 --prelr 0.00083 --beta 0.33 --alpha 7.18--probscale 4.31 --proboffset 4.11 --pt 0.73 --cnprob 0.93 --sum 2.74
```


