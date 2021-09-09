# scPreGAN

## Introduction

The official implementation of "scPreGAN, a deep generative model for 
predicting the response of single cell expression to perturbation".
For reproducing the paper results, please visit XXX.

## Installation

`pip install git+https://github.com/xjwei/scPreGAN.git`

## Example

Example for out of sample prediction data of scRNA-seq data is shown in XXX.
The data utilize in scPreGAN is in the h5ad format. If your data is in other
format, please refer to Scanpy' or anndata's tutorial for how to create one.

The data input to scPreGAN are better the normalized and scaled data, 
you can use follow codes for this purpose.

`sc.pp.filter_genes(adata, min_counts=10)`
`sc.pp.filter_cells(adata, min_counts=3)`
`sc.pp.normalize_per_cell(adata)`
`sc.pp.log1p(adata)`
`sc.pp.scale(adata)`