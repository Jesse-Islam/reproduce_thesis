
import gosip
import os
import gc
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import anndata
import scanpy as sc

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

import shutil
import random
import time
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

import numpy as np
from sklearn.model_selection import train_test_split

def split_and_standard_scale_adata(
    checkpoint,
    validation_ratio: float = 0.1,
    random_state: int = 42
):
    """
    Split an AnnData object into training and validation subsets,
    and apply standard scaling (zero mean, unit variance) per feature
    using only training data.

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
        Standard-scaled training AnnData.
    val_adata
        Standard-scaled validation AnnData.
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
    val_adata   = checkpoint[val_idx, :].copy()
    val_adata.one_hot_labels   = checkpoint.one_hot_labels.iloc[val_idx, :].copy()

    # 3. Compute per-feature mean & std on training data
    X_train = train_adata.X.astype(float)
    # if sparse, convert to dense for mean/std:
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    feature_means = X_train.mean(axis=0, keepdims=True)
    feature_stds  = X_train.std(axis=0, keepdims=True)
    # avoid division by zero
    feature_stds[feature_stds == 0] = 1.0

    # 4. Scale training set
    train_adata.X = (X_train - feature_means) / feature_stds

    # 5. Apply same transform to validation
    X_val = val_adata.X.astype(float)
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()
    val_adata.X = (X_val - feature_means) / feature_stds

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
    '''
    loss_weights = [trial.suggest_float("lambda_adv", 0.99,1.01),
                    trial.suggest_float("lambda_cls", 10,10.01),
                    trial.suggest_float("lambda_rec", 9.99,10.01),
                    trial.suggest_float("lambda_iden", 9.99,10.01),
                   trial.suggest_float("lambda_sink", 9.99,10.01)]
    '''
    loss_weights=[1,1,10]
    dropout_rate = trial.suggest_float('dropout_rate', 0.01,0.25 ,log=False)
    critics=3#trial.suggest_int("critics", 3, 4)
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
    #dataloader,weights = gosip.prepare_data(adata,batch_size=batch_size)
    
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
    dropout_rate = trial.suggest_float('disc_dropout_rate', 0.05,0.25 ,log=False)
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
    dropout_rate_g = trial.suggest_float('gen_dropout_rate', 0.05,0.25,log=False)
    
    latent_dim=trial.suggest_int("latent", latent_min,latent_max)
    beta= float(trial.suggest_int("beta",5 ,10)) # float(trial.suggest_categorical('beta', [1])) float(trial.suggest_categorical('beta', [1]))
    #beta=float(trial.suggest_float("beta", 1.0,1.01))                       

    
    train_adata,val_adata=  split_and_maxabs_scale_adata(adata,validation_ratio)
        
    train_dataloader = gosip.prepare_data(train_adata,num_categories,batch_size=batch_size)
    val_dataloader = gosip.prepare_data(val_adata,num_categories,batch_size=batch_size)

    propagator = gosip.Propagator(input_dim=train_adata.shape[1],
                     num_domains=num_categories,
                     device = device,
                     learning_rate = learning_rate,
                     layer_g = layer_nodes_generator,
                     drpt_g = dropout_rate_g,
                     latent_dim=latent_dim, zi=zi)

      
    loss_fn = gosip.BtcvaeLoss(n_data=train_adata.shape[0], alpha=1.0, beta=beta, gamma=1.0,zi=zi)

    return propagator.train(dataloader=train_dataloader, val_loader=val_dataloader,
                            num_epochs=hyperparam_epochs, burn_in=burn_in, verbose=True,optuna_run=True,
                             trial=trial,loss_fn=loss_fn)

'''
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





def prep_cbnn_obj(trial, features, data, time_var='', event_var='', comp_risk=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Sample hyperparameters from Optuna trial
    ratio = 100  # Adjust the case-base sampling ratio
    dropout = trial.suggest_float('dropout', 0.1, 0.2)  # Dropout rate

    # Sample architecture (number of layers and neurons per layer)
    num_layers = 3
    layers = [trial.suggest_int(f'layer{i}', 10,100) for i in range(num_layers)]  # Number of units per layer

    if isinstance(data, pd.DataFrame):
        data = sample_case_base(data, time=time_var, event=event_var, ratio=ratio, comprisk=comp_risk)

    offset = data['offset']
    offset.reset_index(drop=True, inplace=True)
    data = data.drop(columns=['offset'])

    # Include time_var in features
    feature_columns = [col for col in data.columns if col != event_var]
    input_dim = len(feature_columns)

    # Ensure time_var is included in features
    if time_var not in feature_columns:
        feature_columns.append(time_var)
    
    # Create the model with updated input_dim and the sampled hyperparameters
    model = PrepCbnnModel(input_dim, layers,dropout=dropout).to(device)  # Move model to GPU

    # Ensure time_var is in the features list
    features = [f for f in feature_columns if f != event_var]

    return {
        'network': model,
        'casebaseData': data,
        'offset': offset,
        'timeVar': time_var,
        'eventVar': event_var,
        'features': features
    }
    
# Define the merged Optuna objective function for tuning
def objective(trial, features, data, time_var='', event_var='', comp_risk=False, burn_in=5, patience=5,val_size=0.22222,hyperparam_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Step 1: Prepare the model and data using hyperparameters suggested by Optuna

    random_state=None
    #train_val_data, test_data = train_test_split(train_data, test_size=test_size, random_state=random_state)

    # Split the training+validation data into training and validation sets
    train_data=data.copy()
    train_data, val_data = train_test_split(data, test_size=val_size, random_state=random_state)
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    
    # Get means, standard deviations, and max time
    sds = train_data.apply(np.std)
    means = train_data.apply(np.mean)
    max_time = train_data['time'].max().copy()
    time_var = 'time'
    status_var = 'status'
    
    # Normalize data
    train_data = normalizer(train_data, means, sds, max_time)
    #print(train_data.shape)
    
    val_data = normalizer(val_data, means, sds, max_time)
    #print(val_data.shape)
    features = data.columns[:-2]
    # Prepare CBNN data
    val_data_cb = sample_case_base(val_data, time=time_var, event=status_var, ratio=100, comprisk=False)
    val_data=val_data_cb
    cbnn_prep = prep_cbnn_obj(trial, features, train_data, time_var=time_var, event_var=event_var, comp_risk=comp_risk)
    val_data['offset'] = pd.Series(cbnn_prep['offset'][0].repeat(val_data['offset'].shape[0]))
    # Step 2: Sample additional training-related hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [256,512])  # Tune batch size
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)  # Tune learning rate
    
    # Model, criterion, optimizer, and data loader setup
    model = cbnn_prep['network']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # Assuming MSELoss, adjust if needed

    # Data for training
    #x_train = cbnn_prep['casebaseData'].drop(columns=cbnn_prep['eventVar'])
    x_train = cbnn_prep['casebaseData'][cbnn_prep['features']]
    y_train = cbnn_prep['casebaseData'][cbnn_prep['eventVar']]
    x_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Prepare DataLoader
    offset = torch.tensor(cbnn_prep['offset'].values, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = torch.utils.data.TensorDataset(x_train, y_train, offset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 3: Training loop with validation and Optuna reporting/pruning
    patience_counter = 0
    best_model_wts = None
    best_loss  = float('inf')
    epochs_without_improvement = 0
    #print(train_data.shape)
    for epoch in range(hyperparam_epochs):  # You can limit the max epochs based on trial-suggested value
        start_time = time.time()
        model.train()
        for inputs, targets, offsets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, offset=offsets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation (if val_data is provided)
        if val_data is not None:
            offset_val = torch.tensor(val_data['offset'].values, dtype=torch.float32).unsqueeze(1).to(device)
            #x_val = val_data.drop(columns=cbnn_prep['eventVar'])
            x_val = val_data[cbnn_prep['features']]
            y_val = val_data[cbnn_prep['eventVar']]
            x_val = torch.tensor(x_val.values, dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val, offset=offset_val)
                val_loss = criterion(val_outputs, y_val)

                # Step 4: Report the intermediate results to Optuna
                trial.report(val_loss.item(), epoch)

                # Step 5: Optuna pruning - stop trials that aren't promising
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                # Early stopping logic after burn-in period
                if epoch >= burn_in:
                    if best_loss > val_loss.item():
                        best_loss = val_loss.item()
                        best_model_wts = model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Stop training if no improvement within patience
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        model.load_state_dict(best_model_wts)  # Load best weights
                        return best_loss  # Return the best validation loss

                # Verbose output for tracking
                clear_output(wait=True)
                print(f'Epoch {epoch+1}/{hyperparam_epochs}, Val Loss: {val_loss.item()}, Patience: {patience_counter}')

        # Timing the epoch
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Time per epoch: {epoch_time:.2f} seconds')

    # Load best weights before returning final loss
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return best_loss


def run_study(data, features,hyperparam_epochs=10, time_var='', event_var='', hyperparam_trials=20,min_optuna=10,label="",val_size=None,burn_in=5,patience=5):
    #optuna.study.load_study(study_name=single_cell_data.replace(" ", "_")+"_"+label+"_CBNN.db", storage="sqlite:///"+outdir+"/"+single_cell_data.replace(" ","_")+"_CBNN.db")
    #print(f"Study '{single_cell_data.replace(" ", "_")+"_"+label+"_CBNN.db"}' already exists. Deleting and creating a new one.")
    #optuna.delete_study(study_name=single_cell_data.replace(" ", "_")+"_"+label+"_CBNN.db", storage="sqlite:///"+outdir+"/"+single_cell_data.replace(" ","_")+"_CBNN.db")

    study = optuna.create_study(storage="sqlite:///"+outdir+"/"+single_cell_data.replace(" ","_")+"_CBNN.db",
                            direction='minimize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.HyperbandPruner(min_resource=min_optuna,
                                                                  max_resource=hyperparam_epochs,
                                                                  reduction_factor=5),
                            study_name=single_cell_data.replace(" ", "_")+"_"+label+"_CBNN.db",
                            load_if_exists=True)
    start_time=time.time()
    if len(study.get_trials())<hyperparam_trials:
        study.optimize(lambda trial: objective(trial, features, data, time_var=time_var, event_var=event_var,val_size=val_size,hyperparam_epochs=hyperparam_epochs,burn_in=burn_in,patience=patience), n_trials=hyperparam_trials-len(study.get_trials()))
    print(time.time()-start_time)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    return study



def train_and_evaluate_cbnn(data,specific_metric,num_selected,test_data,hyperparam_trials,hyperparam_epochs=10,n_iterations=10,label="",val_size=0.375,burn_in=4,patience=5,start_time=0.05,end_time=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Split data
    data=data.copy()
    data_og=data.copy()
    train_data=data.copy()
    test_data_og=test_data.copy()
    train_data.reset_index(drop=True, inplace=True)
    # Split the data into training+validation and test sets


    random_state=None

    

    
    # Prepare test data (removing status column)
    features = data.columns[:-2].copy()

    time_var = 'time'
    status_var = 'status'

    
    study=run_study(data, features, time_var=time_var, event_var=status_var,hyperparam_trials=hyperparam_trials,
                    label=label,val_size=val_size,hyperparam_epochs=hyperparam_epochs,burn_in=burn_in,patience=patience)
    best_params=study.best_params

    cbnn_hyperparam = {
        'batch_size': best_params['batch_size'],
        'learning_rate': best_params['learning_rate'],
    
        'layer_nodes': [
            best_params["layer0"],
            best_params["layer1"],
            best_params["layer2"]
        ],
        'dropout_rate': best_params['dropout']
    }



    # Initialize empty lists to store score_cbnn from each iteration
    score_cbnn_list = []
    score_kms=[]
    ipa_scores=[]
    times_list = []
    
    # Define number of iterations
    n_iterations = n_iterations  # Set this value as per your requirement
    data=data_og.copy()
    train_data, val_data = train_test_split(data, test_size=val_size, random_state=random_state)  # Change random_state each iteration
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data=test_data_og.copy()
    test_data.reset_index(drop=True, inplace=True)
    # Get means, standard deviations, and max time

    common_times_og = np.linspace(max(data['time'].min(),test_data['time'].min())+1,min(data['time'].max(),test_data['time'].max())-1, num=10)
    
    #print("got shared_common times")
    shap_values_list=[]
    # Calculate SHAP values

    for i in range(n_iterations):
        # Split the training+validation data into training and validation sets
        data=data_og.copy()
        train_data, val_data = train_test_split(data, test_size=val_size, random_state=random_state)  # Change random_state each iteration
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        test_data=test_data_og.copy()
        test_data.reset_index(drop=True, inplace=True)
    
        # Get means, standard deviations, and max time
        sds = train_data.apply(np.std)
        means = train_data.apply(np.mean)
        max_time = train_data['time'].max().copy()
    
        # Normalize data
        train_data = normalizer(train_data, means, sds, max_time)
        
        val_data = normalizer(val_data, means, sds, max_time)
        test_data = normalizer(test_data, means, sds, max_time)
        test = test_data.drop(columns=[status_var])
        #print("calculating specific times")
        common_times= np.linspace(start_time,end_time, num=10)#(common_times_og.copy())/max_time.copy()
        # Prepare CBNN data
        cbnn_prep_val = sample_case_base(val_data, time=time_var, event=status_var, ratio=100, comprisk=False)
        cbnn_prep = prep_cbnn(features, train_data, time_var=time_var, event_var=status_var, ratio=100, comp_risk=False,
                              layers=[cbnn_hyperparam['layer_nodes'][0],
                                      cbnn_hyperparam['layer_nodes'][1],
                                      cbnn_hyperparam['layer_nodes'][2]],
                              dropout=cbnn_hyperparam['dropout_rate'],
                              lr=cbnn_hyperparam['learning_rate'])
    
        cbnn_prep_val['offset'] = pd.Series(cbnn_prep['offset'][0].repeat(cbnn_prep_val['offset'].shape[0]))
    
        # Fit hazard model
        fit = fit_hazard(cbnn_prep, epochs=2000, batch_size=cbnn_hyperparam['batch_size'], val_data=cbnn_prep_val,burn_in=burn_in,patience=patience)
        #list(np.arange(test_data['time'].min() + 0.0000001,min(test_data['time'].max(), 1),0.05))
        # Compute cumulative incidences and survival estimates
       
        cumulative_incidences, times = cu_inc_cbnn(fit, times=common_times,
                                                   x_test=test_data[list(features) + [time_var]])
        survival_estimates_cbnn = 1 - cumulative_incidences
        # Convert to structured array
        train_val = pd.concat([train_data, val_data], axis=0)
        train_surv = np.array([(e, t) for e, t in zip(train_val['status'], train_val['time'])], dtype=[('event', 'bool'), ('time', 'f8')])
        test_surv = np.array([(e, t) for e, t in zip(test_data['status'], test_data['time'])], dtype=[('event', 'bool'), ('time', 'f8')])
        #print("calculating brier_score")
        # Calculate Brier scores using the interpolated score_cbnn
        _, score_cbnn = brier_score(train_surv, test_surv, survival_estimates_cbnn, common_times)
        # Interpolate the score_cbnn values to match the predefined common_times
        #interp_fn = interp1d(times, score_cbnn, kind='nearest', fill_value="extrapolate")
        #score_cbnn_interp = interp_fn(common_times)
        # Store the Brier scores for this iteration
        score_cbnn_list.append(score_cbnn)
        
        kmf = KaplanMeierFitter()
        kmf.fit(durations=train_surv['time'], event_observed=train_surv['event'])
       
        km_preds=kmf.survival_function_at_times(common_times).to_numpy()
        #print("calculating KM")
        km_preds=np.tile(km_preds, (test_surv.shape[0], 1))
        # Compute Brier scores for Kaplan-Meier model
        times, score_km = brier_score(train_surv, test_surv, km_preds, times)
        #interp_fn = interp1d(times, score_km, kind='nearest', fill_value="extrapolate")
        score_km = score_km#interp_fn(common_times)
        score_kms.append(score_km)
        times = common_times  # Assume times are the same across iterations (since they should be from the same range)
        #print("calculating IPA")
        ipa_cbnn=1-(score_cbnn/score_km)
        ipa_scores.append(ipa_cbnn)
        ###################################
        ###shap_values_list required
        ####################################

    
        #shap_values_list.append([np.random.rand(len(list(features) + [time_var]))])    
        #shap_values_list.append(shap_cbnn(fit, common_times, train_val, x_test=test_data[list(features) + [time_var]]))  # Store the SHAP values

        


    # After all iterations, ensure that times are aligned
    #print(shap_values[0].mean(axis=0).shape)
    
    score_cbnn_list = np.array(score_cbnn_list)
    ipa_scores =np.array(ipa_scores)
    score_kms =np.array(score_kms)
    
    # Stack them into a 3D array of shape (m, n, p)
    
    ###################################
    ###shap_values_list required
    ####################################
    stacked_arrays = np.stack(shap_values_list, axis=0) 
    # Compute the mean and standard deviation along axis 0 (the m axis)
    mean_shap = pd.DataFrame(np.mean(stacked_arrays, axis=0))
    std_shap = pd.DataFrame(np.std(stacked_arrays, axis=0))
    mean_shap.columns=fit['features']
    std_shap.columns=fit['features']
    print("mean.shap columns: ")
    print(mean_shap.columns)
    mean_shap.index=common_times * max_time
    std_shap.index=common_times * max_time
    #print("stacked array shape: "+str(stacked_arrays.shape))
    

    # Compute the mean and standard error of score_cbnn across iterations
    mean_score_cbnn = np.mean(score_cbnn_list, axis=0)
    std_err_score_cbnn = np.std(score_cbnn_list, axis=0,ddof=1) / np.sqrt(n_iterations)
    ipa_mean_score_cbnn = np.mean(ipa_scores, axis=0)
    ipa_std_err_score_cbnn = np.std(ipa_scores, axis=0,ddof=1) / np.sqrt(n_iterations)
    km_mean = np.mean(score_kms, axis=0)
    km_std_err = np.std(score_kms, axis=0,ddof=1) / np.sqrt(n_iterations)
    # Output mean and std_err
    #mean_score_cbnn, std_err_score_cbnn# Fit Cox proportional hazards model
    
    
    est = CoxPHSurvivalAnalysis(ties="efron").fit(train_val[features], train_surv)
    cox_survs = est.predict_survival_function(test_data[features])
    cox_preds = [fn(times) for fn in cox_survs]
    
    # Calculate Brier scores for Cox model
    times, score_cox = brier_score(train_surv, test_surv, cox_preds, times)
    interp_fn = interp1d(times, score_cox , kind='nearest', fill_value="extrapolate")
    score_cox = interp_fn(common_times)
    # Step 1: Generate Kaplan-Meier Curve
    # Fit Kaplan-Meier model

    

    
    # Plot Brier scores with standard error for mean_score_cbnn
    
    # Times are scaled by max_time
    times = common_times * max_time

    df= pd.DataFrame({
                         'score_cox':score_cox,
                         'brier_mean_score_cbnn':mean_score_cbnn,
                         'brier_std_err_score_cbnn':std_err_score_cbnn,
                         'score_km':km_mean,
                         'score_km_std':km_std_err,    
                         'ipa_mean_score_cbnn':ipa_mean_score_cbnn,
                         'ipa_std_err_score_cbnn':ipa_std_err_score_cbnn   })
    
    df=df.rename(columns=lambda col: col + label)
    df['times']=times
    print(features)
    return df,4,4#,mean_shap,std_shap



def plot_with_confidence_intervals(df_means, df_std_err):
    # Extract time from the index
    time = df_means.index
    
    # Create the plot
    plt.figure(figsize=(12, 8))

    # Loop through each gene (columns in the dataframes)
    for gene in df_means.columns:
        # Extract mean and std_err for the gene
        mean_values = df_means[gene]
        std_err_values = df_std_err[gene]

        # Calculate the 95% confidence intervals
        ci_upper = mean_values + 1.96 * std_err_values
        ci_lower = mean_values - 1.96 * std_err_values

        # Plot the mean line
        plt.plot(time, mean_values, label=gene)

        # Plot the confidence interval (fill between)
        #plt.fill_between(time, ci_lower, ci_upper, alpha=0.3)

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Shapley score')
    plt.title('Gene SHAPley importance over Time with 95% Confidence Intervals')
    plt.legend(loc='best')
    
    # Show the plot
    plt.show()



def plot_cumulative_with_confidence_intervals(df_means, df_std_err):
    # Extract time from the index
    time = df_means.index

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Loop through each gene (columns in the dataframes)
    for gene in df_means.columns:
        # Extract mean and std_err for the gene
        mean_values = df_means[gene]
        std_err_values = df_std_err[gene]

        # Calculate cumulative sums of the means
        cumulative_means = np.cumsum(mean_values)

        # Calculate cumulative variances and convert to standard errors
        cumulative_variances = np.cumsum(std_err_values**2)
        cumulative_std_err = np.sqrt(cumulative_variances)

        # Calculate the 95% confidence intervals for the cumulative sums
        ci_upper = cumulative_means + 1.96 * cumulative_std_err
        ci_lower = cumulative_means - 1.96 * cumulative_std_err

        # Plot the cumulative mean line
        plt.plot(time, cumulative_means, label=gene)

        # Plot the cumulative confidence interval (fill between)
        plt.fill_between(time, ci_lower, ci_upper, alpha=0.3)


    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Shapley score')
    plt.title('Gene SHAPley importance over Time with 95% Confidence Intervals')
    plt.legend(loc='best')

    # Show the plot
    plt.show()

# Assuming df_means and df_std_err are your dataframes
#plot_cumulative_with_confidence_intervals(mean_shap_gosip_out_ecp,std_shap_gosip_out_ecp)



def plot_points_with_errorbars(df_means, df_std_err, ax, title, confidence_level=1.96):
    # Create lists to store the final cumulative means and confidence intervals
    cumulative_means = []
    ci_lower = []
    ci_upper = []
    genes = df_means.columns
    
    # Loop through each gene (columns in the dataframes)
    for gene in genes:
        # Extract mean and std_err for the gene
        mean_values = df_means[gene]
        std_err_values = df_std_err[gene]

        # Calculate cumulative sums of the means
        cumulative_mean = np.cumsum(mean_values)

        # Calculate cumulative variances and convert to standard errors
        cumulative_variance = np.cumsum(std_err_values**2)
        cumulative_std_err = np.sqrt(cumulative_variance)

        # Calculate the final cumulative score and the 95% confidence intervals
        final_mean = cumulative_mean.iloc[-1]  # Getting the last value directly
        ci_upper_temp = final_mean + confidence_level * cumulative_std_err.iloc[-1]
        ci_lower_temp = final_mean - confidence_level * cumulative_std_err.iloc[-1]

        # Store the final values in the lists
        cumulative_means.append(final_mean)
        ci_lower.append(ci_lower_temp)
        ci_upper.append(ci_upper_temp)

    # Convert final means and confidence intervals into numpy arrays
    cumulative_means = np.array(cumulative_means)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    
    # Calculate the error bars as the difference between upper and lower CI
    error_bars = ci_upper - cumulative_means

    # Plot points with error bars
    for i, gene in enumerate(genes):
        color = 'red' if ci_lower[i] <= 0 <= ci_upper[i] else 'blue'  # Red if CI crosses zero
        ax.errorbar(cumulative_means[i], gene, xerr=error_bars[i], fmt='o', color=color, capsize=5, capthick=2, markersize=5)

    # Set title, labels, and grid
    ax.set_title(title)
    ax.set_xlabel("Cumulative SHAPLEY Score")
    ax.set_ylabel("Gene")
    ax.grid(True)






# Define a function to perform the interpolation
def interpolate_on_times(df, common_times):
    # Set 'times' as the index for interpolation
    df = df.set_index('times')
    # Reindex the dataframe to the common 'times' and interpolate missing values
    df_interp = df.reindex(common_times).interpolate(method='linear').reset_index()
    # Rename the index column to 'times' again (it becomes index after reset)
    df_interp.rename(columns={'index': 'times'}, inplace=True)
    return df_interp


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
    sc.pp.scale(adata,zero_center=False)
    return adata.copy()    

'''


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







    