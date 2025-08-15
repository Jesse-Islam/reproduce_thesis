import os

import gosip
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import anndata
import scanpy as sc

import shutil
import random
import time
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

def set_seed(seed: int):
    """
    Set the seed for various libraries to ensure reproducibility.
    Parameters:
    seed (int): The seed value to set.
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set numpy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed for CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure deterministic behavior in PyTorch (if possible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # If using multiprocessing with PyTorch
    torch.multiprocessing.set_start_method('spawn', force=True)

# Example usage:
set_seed(1)  # Replace 42 with the desired seed value

def split_and_maxabs_scale_adata(
    checkpoint,
    validation_ratio: float = 0.1,
    random_state: int = 42
):
    """
    Split an AnnData object into training and validation subsets,
    and apply Max-Absolute scaling (preserving zeros) per feature using only training data.

    Parameters
    ----------
    checkpoint
        Full AnnData (with .X and .one_hot_labels).
    validation_ratio
        Fraction of cells to reserve for validation.
    random_state
        Seed for reproducible splitting.

    Returns
    -------
    train_adata
        MaxAbs-scaled training AnnData.
    val_adata
        MaxAbs-scaled validation AnnData.
    """
    # 1. Split indices
    n_obs = checkpoint.n_obs
    all_idx = np.arange(n_obs)
    train_idx, val_idx = train_test_split(
        all_idx, test_size=validation_ratio, random_state=random_state
    )

    # 2. Subset into new AnnData objects
    train_adata = checkpoint[train_idx, :].copy()
    train_adata.one_hot_labels = checkpoint.one_hot_labels.iloc[train_idx, :].copy()
    val_adata = checkpoint[val_idx, :].copy()
    val_adata.one_hot_labels = checkpoint.one_hot_labels.iloc[val_idx, :].copy()

    # 3. Fit MaxAbsScaler on train
    scaler = MaxAbsScaler()
    X_train = train_adata.X
    X_train_scaled = scaler.fit_transform(X_train)
    train_adata.X = X_train_scaled

    # 4. Apply same scaling to validation
    X_val = val_adata.X
    X_val_scaled = scaler.transform(X_val)
    val_adata.X = X_val_scaled

    return train_adata, val_adata

def split_and_standardize_adata(
    checkpoint,
    validation_ratio: float = 0.1,
    random_state: int = 42
):
    """
    Split an AnnData object into training and validation subsets,
    and apply classic standardization (zero-mean, unit-variance) per feature.

    Parameters
    ----------
    checkpoint
        Full AnnData (with .X and .one_hot_labels).
    validation_ratio
        Fraction of cells to reserve for validation.
    random_state
        Seed for reproducible splitting.

    Returns
    -------
    train_adata
        Standardized training AnnData.
    val_adata
        Standardized validation AnnData.
    """
    # 1. Split indices
    n_obs = checkpoint.n_obs
    all_idx = np.arange(n_obs)
    train_idx, val_idx = train_test_split(
        all_idx, test_size=validation_ratio, random_state=random_state
    )

    # 2. Subset into new AnnData objects
    train_adata = checkpoint[train_idx, :].copy()
    train_adata.one_hot_labels = checkpoint.one_hot_labels.iloc[train_idx, :].copy()
    val_adata = checkpoint[val_idx, :].copy()
    val_adata.one_hot_labels = checkpoint.one_hot_labels.iloc[val_idx, :].copy()

    # 3. Compute mean & std on train
    X_train = train_adata.X.astype(float)
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    # guard against zero‐variance features
    feature_std[feature_std == 0] = 1.0

    # 4. Standardize train: (x - mean) / std
    X_train = (X_train - feature_mean) / feature_std
    train_adata.X = X_train

    # 5. Standardize validation with same parameters
    X_val = val_adata.X.astype(float)
    X_val = (X_val - feature_mean) / feature_std
    val_adata.X = X_val

    return train_adata, val_adata


def opt_objective_stargan(trial,adata,num_categories,hyperparam_epochs,validation_ratio=0.1,lambda_adv=1,lambda_cls=1,lambda_rec=10,lambda_iden=1,min_layer_size=750,max_layer_size=1000,bsize=1024,burn_in=10,zi=True):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [bsize])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3,log=True)
    layer_nodes_generator = [trial.suggest_int("gen_layer0", min_layer_size, max_layer_size),
                             trial.suggest_int("gen_layer1", min_layer_size, max_layer_size),
                             trial.suggest_int("gen_layer2", min_layer_size, max_layer_size)]
    layer_nodes_discriminator =[trial.suggest_int("disc_layer0", min_layer_size, max_layer_size),
                             trial.suggest_int("disc_layer1", min_layer_size, max_layer_size),
                             trial.suggest_int("disc_layer2", min_layer_size, max_layer_size)]

    loss_weights=[1,1,10,10,10]
    dropout_rate = trial.suggest_float('dropout_rate', 0.001,0.1 ,log=True)
    critics=3#trial.suggest_int("critics", 1, 1)
    stargan = gosip.StarGAN(input_dim=adata.shape[1],
                     num_domains=[adata.one_hot_labels.shape[1]],
                      device=device,
                      learning_rate=learning_rate,
                      layer_g=layer_nodes_generator,
                      layer_d=layer_nodes_discriminator,
                      lambda_adv=loss_weights[0],
                      lambda_cls=loss_weights[1],
                      lambda_rec=loss_weights[2],
                      #lambda_iden=loss_weights[3],
                      #lambda_sink=loss_weights[4],
                     critics=critics,
                     dropout_rate=dropout_rate,zi=zi)
    train_adata,val_adata=  split_and_maxabs_scale_adata(adata,validation_ratio)


    train_dataloader = gosip.prepare_data(train_adata,num_categories,batch_size=batch_size)
    val_dataloader = gosip.prepare_data(val_adata,num_categories,batch_size=batch_size)
    # Prepare data loader
    #permutation = torch.randperm(labels.size(0))
    #real_data=real_data[permutation]
    #labels=labels[permutation]
    #dataloader,weights = prepare_data(adata,batch_size=batch_size)
    return stargan.train(dataloader=train_dataloader,val_loader=val_dataloader, num_epochs=hyperparam_epochs, burn_in=burn_in,optuna_run=True,trial=trial,verbose=True)


def opt_objective_oracle(trial,adata,num_categories,hyperparam_epochs,validation_ratio=0.1,min_layer_size=750,max_layer_size=1000,bsize=1024,burn_in=10):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [bsize])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3,log=True)
    layer_nodes =[trial.suggest_int("disc_layer0", min_layer_size, max_layer_size),
                             trial.suggest_int("disc_layer1", min_layer_size, max_layer_size),
                             trial.suggest_int("disc_layer2", min_layer_size, max_layer_size)]
    dropout_rate = trial.suggest_float('disc_dropout_rate', 0.001,0.1 ,log=True)
    oracle = gosip.Oracle(input_dim=adata.shape[1],
                     num_domains=num_categories,
                      device=device,
                      learning_rate=learning_rate,
                      layer_d=layer_nodes,
                      drpt_d=dropout_rate)
    
    train_adata,val_adata=  split_and_maxabs_scale_adata(adata,validation_ratio)
    train_dataloader = gosip.prepare_data(train_adata,num_categories,batch_size=batch_size)
    val_dataloader = gosip.prepare_data(val_adata,num_categories,batch_size=batch_size)

    return oracle.train(dataloader=train_dataloader, val_loader=val_dataloader, num_epochs=hyperparam_epochs, burn_in=burn_in, verbose=True,optuna_run=True,trial=trial)



def opt_objective_propagator(trial,adata,num_categories,hyperparam_epochs,validation_ratio=0.1,min_layer_size=750,max_layer_size=1000,bsize=1024,latent_min=5,latent_max=15,burn_in=10,zi=True):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device='cpu'
    # Define the hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [bsize])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3,log=True)
    layer_nodes_generator = [trial.suggest_int("gen_layer0", min_layer_size, max_layer_size),
                             trial.suggest_int("gen_layer1", min_layer_size, max_layer_size),
                             trial.suggest_int("gen_layer2", min_layer_size, max_layer_size)]
    dropout_rate_g = trial.suggest_float('gen_dropout_rate', 0.001,0.1 ,log=True)
    
    latent_dim=trial.suggest_int("latent", latent_min,latent_max)
    beta=float(trial.suggest_categorical('beta', [1]))
                              

    
    # Assume 'adata' is your AnnData object
    indices = np.arange(adata.n_obs)
    
    train_adata,val_adata=  split_and_maxabs_scale_adata(adata,validation_ratio)
    train_dataloader = gosip.prepare_data(train_adata,num_categories,batch_size=batch_size)
    val_dataloader = gosip.prepare_data(val_adata,num_categories,batch_size=batch_size)

    propagator = gosip.Propagator(input_dim=train_adata.shape[1],
                     num_domains=num_categories,
                     device = device,
                     learning_rate = learning_rate,
                     layer_g = layer_nodes_generator,
                     drpt_g = dropout_rate_g,
                     latent_dim=latent_dim,zi=zi)

    
    #loss_fn = gosip.BetaVaeLoss(beta=beta)
    loss_fn =gosip.BtcvaeLoss(n_data=train_adata.shape[0],beta=beta,zi=zi)
    return propagator.train(dataloader=train_dataloader, val_loader=val_dataloader,
                            num_epochs=hyperparam_epochs, burn_in=burn_in, verbose=True,optuna_run=True,
                             trial=trial,loss_fn=loss_fn)


def extract_rank_genes_groups(adata, key='hi', n_genes=None):
    """
    Extracts the ranked genes from the `rank_genes_groups` results stored in adata.uns 
    and returns them as a pandas DataFrame.
    
    Parameters:
    - adata: AnnData object containing the single-cell data.
    - key: The key under which the rank_genes_groups results are stored (default is 'hi').
    - n_genes: Number of top genes to extract for each group. If None, all genes are extracted.
    
    Returns:
    - A pandas DataFrame with the ranked genes, log fold changes, p-values, adjusted p-values, 
      and the group (cluster) information.
    """
    
    # Extract the results from the adata.uns dictionary
    rank_genes = adata.uns[key]
    
    # Get the groups (clusters or cell types)
    groups = rank_genes['names'].dtype.names
    
    result_dfs = {}
    
    # Loop through each group and extract relevant statistics
    for group in groups:
        # Extract the top `n_genes` if specified, otherwise take all genes
        if n_genes is not None:
            df = pd.DataFrame({
                'gene': rank_genes['names'][group][:n_genes],
                'scores_'+ key: rank_genes['scores'][group][:n_genes],
                'logfoldchange': rank_genes['logfoldchanges'][group][:n_genes],
                'pval': rank_genes['pvals'][group][:n_genes],
                'pval_adj': rank_genes['pvals_adj'][group][:n_genes],
            })
        else:
            df = pd.DataFrame({
                'gene': rank_genes['names'][group],
                'scores_'+ key: rank_genes['scores'][group],
                'logfoldchange': rank_genes['logfoldchanges'][group],
                'pval': rank_genes['pvals'][group],
                'pval_adj': rank_genes['pvals_adj'][group],
            })
        
        result_dfs[group] = df

    # Concatenate the results from all groups into a single DataFrame
    final_df = pd.concat([df.assign(group=group) for group, df in result_dfs.items()], ignore_index=True)
    
    return final_df





def pre_process(adata,top_genes=5000):
    print("begin preprocessing...")
    print("filtering...")
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    # Normalizing to median total counts
    print("log(x+1) normalization...")
    #sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)
    print("keeping highly variable...")
    sc.pp.highly_variable_genes(adata, n_top_genes=top_genes)
    sc.pl.highly_variable_genes(adata)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.normalize_total(adata, target_sum=1e4)
    #print("z-scaling...")
    sc.pp.scale(adata)
    return adata.copy()    

    