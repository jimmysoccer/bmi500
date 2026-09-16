import time
import cProfile

# %%
start_time = time.time()

from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
import scanpy as sc

import sys
import argparse

print(f"[PROFILE] Imports: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
# sc.settings.n_jobs = int(sys.argv[4])
sc.settings.n_jobs = 1

print(f"using {sc.settings.n_jobs} threads")

print(f"[PROFILE] Settings: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--data-dir', type=str, help='Directory containing the dataset subdirectories', default='data')
parser.add_argument('--data-set', type=str, help='Dataset name, which is the subdirectory name', default='pbmc3k')
parser.add_argument('--out-dir', type=str, help='Output directory', required=False, default='data')
parser.add_argument('--num-threads', type=int, help='Number of threads', default=1, required=False)

args = parser.parse_args()

datadir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
dataset = args.data_set
outdir = args.out_dir if args.out_dir.endswith('/') else args.out_dir + '/'
nthreads = args.num_threads

print(f"[PROFILE] Argument parsing: {time.time() - start_time:.6f} seconds")


#%%
start_time = time.time()

# I/O
results_file = "/".join([outdir, dataset + '.scanpy.h5ad'])

adata = sc.read_10x_mtx(
    "/".join([datadir, dataset, 'filtered_gene_bc_matrices']),
    var_names='gene_symbols',
    cache=True)

adata.var_names_make_unique()

print(f"[PROFILE] Data loading: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

# preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

print(f"[PROFILE] Basic filtering: {time.time() - start_time:.6f} seconds")


#%%
start_time = time.time()

# metric / normalization
#adata.var['mt'] = adata.var_names.str.startswith('MT-')
#sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

#adata = adata[adata.obs.n_genes_by_counts < 2500, :]
#adata = adata[adata.obs.pct_counts_mt < 5, :]

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print(f"[PROFILE] Normalization: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

# highly variable genes
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat",
    n_top_genes=2000
)

adata.raw = adata
adata = adata[:, adata.var.highly_variable]

print(f"[PROFILE] Highly variable genes: {time.time() - start_time:.6f} seconds")


#%%
start_time = time.time()

# regress out effects / scaling
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata)

print(f"[PROFILE] Scaling: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

# report adata
# adata.write(results_file)
# adata

print(f"[PROFILE] Report adata section: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

# pca
sc.tl.pca(adata, svd_solver='arpack', n_comps=30)

# adata.write(results_file)
# adata

print(f"[PROFILE] PCA: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

# neighborhood graph
sc.pp.neighbors(adata, n_pcs=30)

print(f"[PROFILE] Neighbors: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

# for fixing disconnected clusters or connectivity issues:
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)
#cs.tl.umap(adata, init_pos='paga')

# adata.write(results_file)
# adata

print(f"[PROFILE] PAGA/commented section: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

# clustering
#sc.tl.leiden(adata)
sc.tl.louvain(adata, resolution=0.5)

print(f"[PROFILE] Louvain clustering: {time.time() - start_time:.6f} seconds")


#%%
start_time = time.time()

# umap
sc.tl.umap(adata, n_components=30)

print(f"[PROFILE] UMAP: {time.time() - start_time:.6f} seconds")


#%%
start_time = time.time()

adata.write(results_file)
adata

print(f"[PROFILE] Write output: {time.time() - start_time:.6f} seconds")


# %%
start_time = time.time()

profile_file = f"rank_genes_{dataset}_profile.prof"
cProfile.run(
    "sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', use_raw=True)",
    profile_file
)

print(f"[PROFILE] Rank genes: {time.time() - start_time:.6f} seconds")