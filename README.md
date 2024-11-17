# gemGAT
This is the official repo of gemGAT: **Cross-tissue Graph Attention Networks for Semi-supervised Gene Expression Prediction**. gemGAT aims to enhance gene expression prediction across different tissues.

## Dependencies

The model is trained on NVIDIA GeForce RTX 3090. Here are dependencies in Python. Note that you may upgrade those packages to fit your data and experimental settings.

`pytorch`: 1.13.0

`dgl-cuda11.6`: 0.9.1

`numpy`: 1.23.4

`pandas`: 1.4.2

## Dataset
gemGAT requires the following dataset to train the model:

1. Gene expression in the source tissue.

2. Gene-gene network (e.g., co-expression network) in both source and the target tissue.

A sample dataset can be found [here](https://drive.google.com/drive/folders/1z_qdChCJM3GdjBTQfQWKjuK7oyF3oXir?usp=drive_link) to illustrate the data format allowed by the program, in which we have four files corrsponding to tissue `Brain Amygdala` processed from [ADNI](https://adni.loni.usc.edu/) dataset:

`expr_in_Brain-Amygdalaadni.csv`: This csv file saves gene expression data in the source tissue. The first row and the first column are subject and gene ID, respectively. Each element corresponds to gene expression regarding a specific subject and a specific gene.

`expr_out_Brain-Amygdalaadni.csv`: This csv file saves gene expression data in the target tissue for training purpose. The first row and the first column are the same set of subject ID and (usually more) gene ID, respectively.

`graph_in_Brain-Amygdalaadni.csv`: This csv file saves gene-gene netnetwork in the source tissue. The first row and the first column are both IDs for the same set of genes in the same order. Gene-gene network is a binary matrix that indicates interactions between genes, and can be any known gene-gene networks or constructed via existing tools, such as co-expression network constructed by [WGCNA](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-559). We constructed our gene-gene co-expression networks of both source and target tissues via WGCNA using gene expression data (e.g., `expr_in_Brain-Amygdalaadni.csv` and `expr_out_Brain-Amygdalaadni.csv`).

`graph_out_Brain-Amygdalaadni.csv`: This csv file saves gene-gene netnetwork in the target tissue. The first row and the first column are both IDs for the same set of genes in the same order. Note that genes of the source tissue are covered by those of the target tissue. We order genes in the source tissue before genes that are in the target tissue but not in the source tissue.

## Training

## testing
