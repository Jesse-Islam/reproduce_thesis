from sklearn.model_selection import train_test_split

from cbnn import sample_case_base
    
import os
import gc
import re
import gosip
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

def process_pre_split_adatas(train_adata,
                             val_adata,
                             time_var='time',
                             event_var='status',
                             weight_var=None,
                             ratio=100,
                             pre_cb_data=False):
    """
    Given two AnnDatas (already split and with X pre-normalized), 
    compute inverse-probability weights, scale time, sample case-base,
    and return (df_train, df_val, max_time, means, sds[, debug_dict]).
    """
    import pandas as pd
    import numpy as np

    def _build_df(adata):
        # Features are assumed already normalized in adata.X
        df = pd.DataFrame(adata.X, columns=adata.var_names)
        obs = adata.obs
        df['time']   = obs[time_var].values
        df['status'] = obs[event_var].values
        if weight_var is None:
            df['to_weigh'] = 'uniform'
        else:
            df['to_weigh'] = obs[weight_var].astype(str).tolist()
        return df

    # 1) Build raw DataFrames
    train_df = _build_df(train_adata)
    val_df   = _build_df(val_adata)

    # 2) Compute inv_weights on TRAIN
    train_n = len(train_df)
    train_sizes = train_df.groupby('to_weigh').size()
    G_train = len(train_sizes)
    train_wmap = {g: train_n/(G_train * sz) for g, sz in train_sizes.items()}
    train_df['inv_weight'] = train_df['to_weigh'].map(train_wmap)

    # 3) Compute inv_weights on VAL
    val_n = len(val_df)
    val_sizes = val_df.groupby('to_weigh').size()
    G_val = len(val_sizes)
    val_wmap = {g: val_n/(G_val * sz) for g, sz in val_sizes.items()}
    val_df['inv_weight'] = val_df['to_weigh'].map(val_wmap)

    # 4) Drop helper column
    train_df.drop(columns=['to_weigh'], inplace=True)
    val_df.drop(columns=['to_weigh'], inplace=True)

    # 5) Compute normalization stats (for transparency/callbacks)
    sds       = train_df.apply(np.std)
    means     = train_df.apply(np.mean)
    max_time  = train_df['time'].max()

    # 6) Apply only time-scaling & preserve inv_weight/status
    def normalizer_weighted(df):
        out = df.copy()
        out['status']     = df['status']
        out['time']       = df['time'] / max_time
        out['inv_weight'] = df['inv_weight']
        return out

    train_df = normalizer_weighted(train_df)
    val_df   = normalizer_weighted(val_df)

    # 7) Sample case–base
    def sample_cb(df):
        df = df.reset_index(drop=True)
        return sample_case_base(df, time='time', event='status', ratio=ratio)

    df_train = sample_cb(train_df).reset_index(drop=True)
    df_val   = sample_cb(val_df).reset_index(drop=True)

    if pre_cb_data:
        debug = {'pre_cb_train': train_df, 'pre_cb_val': val_df}
        return df_train, df_val, max_time, means, sds, debug

    return df_train, df_val, max_time, means, sds
def normalizer_weighted(data, means, sds, max_time):
    normalized = data.copy()
    for col in data.columns:
        if col in ['status', 'time', 'inv_weight']:
            continue
        if len(data[col].unique()) > 1:
            normalized[col] = (data[col] - means[col]) / sds[col] 
    normalized['status'] = data['status']
    normalized['time'] = data['time'] / max_time
    normalized['inv_weight'] = data['inv_weight']  # preserve weights as is

    return normalized.copy()
    
def split_and_scale_time(adata, time_var='time', event_var="status", weight_var=None, 
                         val_size=0.2, ratio=100, random_state=None,pre_cb_data=False):

    # Convert AnnData to DataFrame
    X = pd.DataFrame(adata.X, columns=adata.var_names)
    obs = adata.obs.copy()

    # Combine features and annotations
    full_df = X.copy()
    full_df["time"] = obs[time_var].values
    full_df["status"] = obs[event_var].values
    if weight_var is None:
        full_df["to_weigh"] = "uniform"  # single group for uniform weights
    else:
        full_df["to_weigh"] = obs[weight_var].astype(str).tolist()

    # Train/validation split
    train_df, val_df = train_test_split(full_df, test_size=val_size, random_state=random_state)
    # --- Compute weights for training set ---
    train_total = len(train_df)
    # Compute group sizes for only those groups present in train_df
    train_group_sizes = train_df.groupby("to_weigh").size()
    num_train_groups = len(train_group_sizes)
    # Build a mapping from each group to its weight
    train_weight_map = {grp: train_total / (num_train_groups * size) for grp, size in train_group_sizes.items()}
    # Map the weights back to train_df
    train_df["inv_weight"] = train_df["to_weigh"].map(train_weight_map)

    # --- Compute weights for validation set ---
    val_total = len(val_df)
    val_group_sizes = val_df.groupby("to_weigh").size()
    num_val_groups = len(val_group_sizes)
    val_weight_map = {grp: val_total / (num_val_groups * size) for grp, size in val_group_sizes.items()}
    val_df["inv_weight"] = val_df["to_weigh"].map(val_weight_map)

    # Remove the original weight variable (no longer needed)
    train_df.drop(columns=["to_weigh"], inplace=True)
    val_df.drop(columns=["to_weigh"], inplace=True)

    # Calculate normalization statistics from the training set
    sds = train_df.apply(np.std)
    means = train_df.apply(np.mean)
    max_time = train_df["time"].max().copy()

    # Normalize the data (using a normalizer that leaves inv_weight unchanged)
    train_df = normalizer_weighted(train_df, means, sds, max_time)
    val_df = normalizer_weighted(val_df, means, sds, max_time)

    # Sampling case-base from the normalized data
    def sample_case_base_from_anndata(df, ratio=100):
        df = df.reset_index(drop=True)
        cb_df = sample_case_base(df, time="time", event="status", ratio=ratio)
        return cb_df

    df_train = sample_case_base_from_anndata(train_df, ratio).copy()
    df_val = sample_case_base_from_anndata(val_df, ratio).copy()

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    if pre_cb_data:
        return df_train, df_val, max_time, means, sds, {"pre_cb_train":train_df,"pre_cb_val":val_df}
    return df_train, df_val, max_time, means, sds

def prep_cbnn(features, data, offset=np.nan, time_var='', event_var='', ratio=100, comp_risk=False, 
              optimizer=None, layers=[10, 10], dropout=None, lr=0.001, device='cpu'):

    offset = data['offset']
    offset.reset_index(drop=True, inplace=True)

    # Extract inverse weights
    sample_weights = data['inv_weight'].reset_index(drop=True).copy()
    # Drop offset and inv_weight from features
    data = data.drop(columns=['offset', 'inv_weight'])

    feature_columns = [col for col in data.columns if col != event_var]
    if time_var not in feature_columns:
        feature_columns.append(time_var)

    model = PrepCbnnModel(len(feature_columns), layers, dropout=dropout).to(device)

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    features = [f for f in feature_columns if f != event_var]

    return {
        'network': model,
        'casebaseData': data,
        'offset': offset,
        'timeVar': time_var,
        'eventVar': event_var,
        'features': features,
        'sample_weights': sample_weights,
        'optimizer' : optimizer
    }

def fit_hazard_optuna(cbnn_prep, 
                      epochs=2000, 
                      batch_size=1024, 
                      val_data=None, 
                      patience=5, 
                      burn_in=0, 
                      trial=None, 
                      use_optuna=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    offset = torch.tensor(cbnn_prep['offset'].values, dtype=torch.float32).unsqueeze(1).to(device)
    x_train = cbnn_prep['casebaseData'][cbnn_prep['features']]
    y_train = cbnn_prep['casebaseData'][cbnn_prep['eventVar']]
    sample_weights = torch.tensor(cbnn_prep['sample_weights'].values, dtype=torch.float32).unsqueeze(1).to(device)

    x_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = torch.utils.data.TensorDataset(x_train, y_train, offset, sample_weights)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = cbnn_prep['network'].to(device)
    criterion = nn.BCELoss(reduction='none')
    optimizer = cbnn_prep['optimizer']

    best_loss = float('inf')
    best_model_wts = None
    patience_counter = 0
    val_weights = torch.tensor(val_data['inv_weight'].values, dtype=torch.float32).unsqueeze(1).to(device)
    offset_val = torch.tensor(val_data['offset'].values, dtype=torch.float32).unsqueeze(1).to(device)
    x_val = val_data[cbnn_prep['features']]
    y_val = val_data[cbnn_prep['eventVar']]
    x_val = torch.tensor(x_val.values, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for inputs, targets, offsets, weights in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, offset=offsets)
            loss = criterion(outputs, targets)
            weighted_loss = (loss * weights).mean()
            weighted_loss.backward()
            optimizer.step()

        # Validation
        if val_data is not None:


            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val, offset=offset_val)
                val_loss_unweighted = criterion(val_outputs, y_val)
                val_loss = (val_loss_unweighted * val_weights).mean()

            if use_optuna and trial is not None:
                trial.report(val_loss.item(), step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if epoch >= burn_in:
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_model_wts = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            clear_output(wait=True)
            print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss.item():.4f} | Patience: {patience_counter}")

        print(f"Epoch {epoch+1} took {time.time() - start_time:.2f} sec")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    cbnn_prep['resultOfFit'] = {
        'model': model,
        'loss': best_loss
    }
    cbnn_prep['xTrain'] = x_train
    cbnn_prep['yTrain'] = y_train

    return cbnn_prep


class PrepCbnnModel(nn.Module):
    def __init__(self, input_dim, layers,dropout=0.1):
        super(PrepCbnnModel, self).__init__()
        
        # Define the layers
        # Dynamically define the layers
        fc_layers = []
        
        # First layer
        fc_layers.append(nn.Linear(input_dim, layers[0]))
        fc_layers.append(nn.LeakyReLU())
        #fc_layers.append(nn.BatchNorm1d(layers[0])) 
        if dropout is not None:
            fc_layers.append(nn.Dropout(dropout))
        
        # Middle layers
        for i in range(len(layers) - 1):
            fc_layers.append(nn.Linear(layers[i], layers[i + 1]))
            fc_layers.append(nn.LeakyReLU())
            #fc_layers.append(nn.BatchNorm1d(layers[i + 1])) 
            if dropout is not None:
                fc_layers.append(nn.Dropout(dropout))
        
        # Final layer
        fc_layers.append(nn.Linear(layers[-1], 1))
        
        # Create the sequential model
        self.fc = nn.Sequential(*fc_layers)
        # Initialize weights using Xavier initialization
        #self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                # Apply Xavier initialization
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, offset):
        x = self.fc(x)
        x = torch.add(x, offset)  # add offset with input features
        x = torch.sigmoid(x)
        return x






#########TORCH VERSION

import torch
import numpy as np

def cu_inc_cbnn(fit, times, x_test):
    """
    Predict cumulative incidence curves for each sample using a CBNN.

    Returns:
        - cif_matrix: shape (n_samples, n_timepoints) as torch.Tensor
        - times: shape (n_timepoints,) as torch.Tensor
    """
    # Ensure model is in eval mode and on the right device
    fit['network'].eval()
    device = next(fit['network'].parameters()).device

    # Add dummy time column
    time_column = x_test.shape[1]
    time_col = torch.zeros((x_test.shape[0], 1), device=device)
    x_test_tensor = torch.cat([x_test.to(device), time_col], dim=1)

    # Offset tensor
    temp_offset = torch.zeros((x_test.shape[0], 1), device=device)

    # Initialize result array
    n_samples = x_test.shape[0]
    n_times = len(times)
    results = torch.full((n_times, n_samples + 1), float('nan'), device=device)
    results[:, 0] = torch.tensor(times, device=device)
    
    # Predict for each time point
    for i, t in enumerate(times):
        x_test_tensor[:, time_column] = t
        with torch.no_grad():
            preds = fit['network'](x_test_tensor, temp_offset).squeeze()
            preds = preds.clamp(min=1e-6, max=1 - 1e-6)  # numerical stability
            results[i, 1:] = preds
    delta_ts = torch.tensor(np.diff(times), dtype=torch.float32, device=device)
    delta_ts = torch.cat([delta_ts, delta_ts[-1].unsqueeze(0)])  # keep length = len(times)

    # Compute cumulative incidence
    
    # Compute cumulative hazard using trapezoidal rule
    for j in range(1, results.shape[1]):
        p = results[:, j].clamp(min=1e-6, max=1 - 1e-6)
        hazard = p / (1 - p)
    
        # Average adjacent hazard values for trapezoidal approximation
        hazard_avg = 0.5 * (hazard[1:] + hazard[:-1])
        cum_hazard = torch.zeros_like(hazard)
        cum_hazard[1:] = torch.cumsum(hazard_avg * delta_ts[:-1], dim=0)
    
        results[:, j] = 1 - torch.exp(-cum_hazard)
    # Extract final CIF matrix and times
    times_tensor = results[:, 0]
    cif_matrix = results[:, 1:].T  # shape: (n_samples, n_timepoints)

    return cif_matrix, times_tensor



def auc_cu_inc_cbnn_gosip(fit, times, x_test, return_cif=False):
    """
    Compute AUC (area under cumulative incidence curve) per sample.

    Returns:
        - aucs: shape (n_samples,) as torch.Tensor
        - (optional) cif_matrix: shape (n_samples, n_timepoints)
        - (optional) times: shape (n_timepoints,)
    """
    cif_matrix, times_tensor = cu_inc_cbnn(fit, times, x_test)  # (n_samples, n_timepoints)
    device = cif_matrix.device

    # Trapezoidal AUC computation
    # AUC = ∑ ((f(t_i) + f(t_{i+1}))/2) * Δt
    delta_ts = times_tensor[1:] - times_tensor[:-1]  # (T-1,)
    y_left = cif_matrix[:, :-1]
    y_right = cif_matrix[:, 1:]
    aucs = 0.5 * torch.sum((y_left + y_right) * delta_ts, dim=1)  # (n_samples,)

    if return_cif:
        return aucs.to(device), cif_matrix.to(device), times_tensor.to(device)
    return aucs.to(device)



def cbnn_objective(trial, adata,burn_in, patience,layer_low,layer_high,time_var,event_var,epochs,val_size,weight_var=None):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    num_layers = 3
    layers = [trial.suggest_int(f'layer{i}', layer_low,layer_high) for i in range(num_layers)]  # Number of units per layer
    
    set_estimate=adata.obs[event_var].sum()*100+adata.obs[event_var].sum()
    # Prepare data
    train_adata,val_adata=  split_and_maxabs_scale_adata(adata,val_size)
    cb_df_train, cb_df_val, _, _, _=process_pre_split_adatas(train_adata,
                                                             val_adata,
                                                             time_var=time_var,
                                                             event_var=event_var,
                                                             weight_var=weight_var,
                                                             ratio=100,
                                                             pre_cb_data=False)
    #cb_df_train, cb_df_val, _, _, _ = split_and_scale_time(adata, time_var=time_var, event_var=event_var,weight_var=weight_var)
    features = cb_df_train.columns[:-2].copy()

    # Prepare CBNN
    cbnn_prep = prep_cbnn(features, cb_df_train, time_var="time", event_var="status",
                          ratio=100, comp_risk=False, layers=layers, dropout=dropout, lr=lr)
    batch_size = trial.suggest_categorical("batch_size", [int(set_estimate/100),int(set_estimate/50)])
    # Train and validate with pruning
    cbnn_fit = fit_hazard_optuna(cbnn_prep,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  val_data=cb_df_val,
                                  burn_in=burn_in,
                                  patience=patience,
                                  trial=trial,
                                  use_optuna=True)

    return cbnn_fit['resultOfFit']['loss']


def run_study_cbnn(adata,time_var,event_var,outdir,weight_var=None,layer_low=128,layer_high=256,
              hyperparam_epochs=100,
              hyperparam_trials=20,
              min_optuna=5,
              label="",
              burn_in=20,
              patience=5,
            val_size=0.2):
    """
    Run an Optuna hyperparameter optimization study for the CBNN model.

    Parameters:
        adata: AnnData object
        hyperparam_epochs: Number of epochs per trial
        hyperparam_trials: Total number of Optuna trials
        min_optuna: Minimum resource for Hyperband pruning
        label: Label suffix for the study
        burn_in: Number of epochs to ignore before early stopping kicks in
        patience: Patience for early stopping
    """

    study_name =  label + "_CBNN.db"
    db_path = f"sqlite:///{outdir}/{study_name}"

    # Create or load existing study
    study = optuna.create_study(
        storage=db_path,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min_optuna,
            max_resource=hyperparam_epochs,
            reduction_factor=5
        ),
        study_name=study_name,
        load_if_exists=True
    )

    # Run study
    start_time = time.time()
    n_remaining_trials = hyperparam_trials - len(study.get_trials())
    if n_remaining_trials > 0:
        study.optimize(
            lambda trial: cbnn_objective(trial, adata=adata,burn_in=burn_in,patience=patience,layer_low=layer_low,layer_high=layer_high,event_var=event_var,time_var=time_var,epochs=hyperparam_epochs,val_size=val_size, weight_var=weight_var),
            n_trials=n_remaining_trials
        )

    # Print results
    print(f"Study completed in {time.time() - start_time:.2f} seconds")
    print("Best trial:")
    best = study.best_trial
    print(f"  Value: {best.value}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    return study


def paired_t(
    base_dist_vec: torch.Tensor,
    dist_vec: torch.Tensor,
):
    """
    Computes:
      1) Paired t-statistic for base_dist_vec vs. dist_vec
    
    No SciPy dependency—all on-device in PyTorch.

    Args:
      base_dist_vec: Tensor of shape [N]
      dist_vec:      Tensor of shape [N]

    Returns:
      t_stat:        Scalar tensor with the paired t-statistic.
    """
    # 1) paired t-stat
    diff = base_dist_vec - dist_vec                    # [N]
    mean = diff.mean()                                 # scalar
    std = diff.std(unbiased=True).clamp_min(1e-8)      # avoid zero
    n = diff.size(0)                                   # batch size N
    se = std / torch.sqrt(torch.tensor(
        n, dtype=mean.dtype, device=mean.device
    ))
    t_stat = mean / se

    return t_stat

def full_perturbation_analysis_cbnn(
    start_tensor: torch.Tensor,
    transition_label_tensor: torch.Tensor,
    stargan,
    oracle,
    propagator,
    cbnn,
    num_categories: list[int],
    *,
    lambda_param: float = 0.0,
    times=[0.1,0.2,0.3,0.4,0.5,0.6,0.7],
    propagation: bool = True,
    network_fdr_pval_threshold=0.10
):
    device = stargan.device
    batch  = start_tensor.size(0)
    d      = start_tensor.size(1)

    # ── allocate on device ─────────────────────────────────────────────── 🆕
    violins_t   = torch.empty((d, batch), device=device)   # ← rows = features, cols = samples
    acc_t       = torch.empty(d, device=device)
    change_t    = torch.empty(d, device=device)
    #impact_mat  = torch.zeros((d, d), device=device, dtype=start_tensor.dtype)
    t_stat_t = torch.empty(d, device=device, dtype=torch.float32)   # 1‑D, length d
    t_stats, p_vals = [], []               # will live on CPU after SciPy call
    cbnn_ci_aucs = torch.empty((d, batch), device=device)   # ← rows = features, cols = samples
    t_stat_t_cbnn =torch.empty(d, device=device, dtype=torch.float32)
    # ── baselines (unchanged) ─────────────────────────────────────────────
    td      = start_tensor.to(device).clone()
    p_td    = propagator.G(td) if propagation else td
    gtd     = stargan.G(td, transition_label_tensor)
    #p_gtd   = propagator.G(gtd)
    
    base_logits = oracle.D(p_td if propagation else td)
    base_acc_list, [base_dist_vec] = gosip.calculate_category_total_accuracy(
        base_logits, transition_label_tensor, num_categories
    )
    eps           = torch.finfo(gtd.dtype).eps
    #baseline_dist_per_feature = torch.abs(p_td - p_gtd).clamp_min(eps)
    ##CHANGES##
    #alpha_r = np.arange(1, d+1) * (network_fdr_pval_threshold / d)
    #crit_np  = stats.t.ppf(1 - (alpha_r/2), df=batch-1)              # length-M
    #crit_torch = torch.tensor(crit_np, device=device, dtype=torch.float32)
    ##CHANGES##
    if propagation:
        od_cbnn = auc_cu_inc_cbnn_gosip(cbnn,times,p_td.clone())
    else:
        od_cbnn = auc_cu_inc_cbnn_gosip(cbnn,times,td.clone())

    
    with torch.inference_mode():
        for col in range(d):
            orig_col, goal_col = td[:, col].clone(), gtd[:, col].clone()
            td[:, col] = goal_col.clone()

            p_td_pert          = propagator.G(td) if propagation else td
            p_td_pert[:, col]  = goal_col.clone()

            preds = oracle.D(p_td_pert if propagation else td)
            if propagation:
                pd_cbnn = auc_cu_inc_cbnn_gosip(cbnn,times,p_td_pert.clone().to(device))
            else:
                pd_cbnn = auc_cu_inc_cbnn_gosip(cbnn,times,td.clone().to(device))
            
            cbnn_ci_aucs[col]=gosip.calculate_valid_ratio(od_cbnn.clone(), pd_cbnn.clone())    
            acc_list, [dist_vec] = gosip.calculate_category_total_accuracy(
                preds, transition_label_tensor, num_categories
            )
            
            ### calculate oracle score
            dist_improv  = gosip.calculate_valid_ratio(base_dist_vec, dist_vec)
            violins_t[col] = dist_improv        

            acc_t[col]   = torch.tensor(acc_list[0], device=device) 

            ### calculate t statistics
            t_stat_t[col] = paired_t(
                base_dist_vec=base_dist_vec,
                dist_vec=dist_vec
            )
            t_stat_t_cbnn[col] = paired_t(
                base_dist_vec=od_cbnn,
                dist_vec=pd_cbnn
            )
            change_t[col]= torch.median(goal_col - orig_col)
            ### get the per gene impact
            #pert_dist_per_feature        = torch.abs(p_td_pert - p_gtd).clamp_min(eps)
            #closeness        = gosip.calculate_valid_ratio(baseline_dist_per_feature, pert_dist_per_feature,
            #                                         lambda_param=lambda_param)
            #mean = closeness.mean(axis=0)
            #std  = closeness.std(axis=0,unbiased=True)
            #se   = std / np.sqrt(batch)
            #t = mean / (se + 1e-12)                 # (N,)          

            #keep  = gosip.bh_tstat_torch_precomputed(t, crit_torch)
            #filtered_median = torch.median(closeness, dim=0).values * keep.to(mean.dtype)#torch.median(closeness, dim=0).values
            
            #impact_mat[col]  = filtered_median
            td[:, col] = orig_col
        t_np   = t_stat_t.cpu().numpy()                 # one copy
        df     = batch - 1
        p_np   = 2.0 * stats.t.sf(np.abs(t_np), df)     # vectorised sf; shape (d,)
        
        t_np_cbnn   = t_stat_t_cbnn.cpu().numpy()                 # one copy
        p_np_cbnn   = 2.0 * stats.t.sf(np.abs(t_np_cbnn), df)     # vectorised sf; shape (d,)

    # ── single off‑device copy at the end ────────────────────────────────
    violins_np  = violins_t.cpu().numpy()
    accuracy_np = acc_t.cpu().numpy()
    change_np   = change_t.cpu().numpy()
    #impact_np   = impact_mat.clamp(min=0.0).cpu().numpy()
    t_stat_np   = np.asarray(t_np)
    p_val_np    = np.asarray(p_np)
    t_stat_np_cbnn   = np.asarray(t_np_cbnn)
    p_val_np_cbnn    = np.asarray(p_np_cbnn)    
    cbnn_ci_aucs     =np.asarray(cbnn_ci_aucs.cpu().numpy())
    return violins_np, accuracy_np, change_np, t_stat_np, p_val_np,cbnn_ci_aucs, p_val_np_cbnn# impact_np,cbnn_ci_aucs, p_val_np_cbnn


###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
#UPDATING!!!
def full_report_cbnn(adata,stargan,oracle,propagator,cbnn,
                shared_filter,filter_criteria_start,filter_criteria_goal,
                main_path,outdir,
                top_n=20,percentile=90,alpha=0.5,sample_fraction=1,
                num_categories=None,category_labels=None,device=None,
                *,umap=False,oracle_performance=False,custom_path=None,
                lambda_param=0.0,times=[0.1,0.2,0.3,0.4,0.5,0.6,0.7],
                k=1):

    torch.cuda.empty_cache()
    stargan.D.eval()
    stargan.G.eval()
    oracle.D.eval()
    propagator.G.eval()

    # the user can specify a shared filter, or specify none. if none, the shared_filter will not be considered during contrasts.
    if shared_filter is None:
        shared_adata=adata.copy()
        shared_adata.one_hot_labels=adata.one_hot_labels.copy()
        shared_filter=["None"]
    else:
        shared_adata=adata[adata.one_hot_labels[shared_filter].all(axis=1),:].copy()
        shared_adata.one_hot_labels=adata.one_hot_labels[adata.one_hot_labels[shared_filter].all(axis=1)].copy()
    # create an identifier label.
    if len(shared_filter) >1:
        identifier_label= "shared_filters_"+"__".join(shared_filter)+"_from_"+filter_criteria_start[0]+"_to_"+filter_criteria_goal[0]
    else:
        if shared_filter[0]=="None":
            identifier_label= "from_"+filter_criteria_start[0]+"_to_"+filter_criteria_goal[0]
        else:
            identifier_label= "shared_filters_"+shared_filter[0]+"_from_"+filter_criteria_start[0]+"_to_"+filter_criteria_goal[0]
    identifier_label = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "_", identifier_label)
    result_path=main_path+outdir+"/"+identifier_label+"/"
    if custom_path is not None:
        result_path=main_path+outdir+"/"+custom_path+"/"
    #generate results directory
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # set up prediction mode for all torch components.
    with torch.no_grad():
        #Convert adata.X into start and goal tensors, along with the split adata representing the start and goal groups.
        start_adata,start_state_tensor, start_label_tensor = gosip.process_adata_to_tensors(shared_adata,1, filter_criteria_start)
        goal_adata,goal_state_tensor, goal_label_tensor = gosip.process_adata_to_tensors(shared_adata,1, filter_criteria_goal)
        # place the start one-hot-encodings on the goal locations. 0 out the start one-hot encodings. this creates the ideal "transition" matrix.
        transition_labels=start_adata.one_hot_labels.copy()
        transition_labels[start_adata.one_hot_labels.iloc[:,(~start_adata.one_hot_labels.columns.str.contains(filter_criteria_start[0]) & start_adata.one_hot_labels.columns.str.contains(filter_criteria_goal[0]))].columns] = (start_adata.one_hot_labels.iloc[:,start_adata.one_hot_labels.columns.str.contains(filter_criteria_start[0])])
        transition_labels[(start_adata.one_hot_labels.iloc[:,start_adata.one_hot_labels.columns.str.contains(filter_criteria_start[0])].columns)] = 0
        transition_label_tensor=torch.tensor(transition_labels.values).float()

        # inverse of the previous step, to go from goal to start instead
        inverse_transition_labels=goal_adata.one_hot_labels.copy()
        inverse_transition_labels[goal_adata.one_hot_labels.iloc[:,(~goal_adata.one_hot_labels.columns.str.contains(filter_criteria_goal[0]) & start_adata.one_hot_labels.columns.str.contains(filter_criteria_start[0]))].columns] = (goal_adata.one_hot_labels.iloc[:,goal_adata.one_hot_labels.columns.str.contains(filter_criteria_goal[0])])
        inverse_transition_labels[(goal_adata.one_hot_labels.iloc[:,goal_adata.one_hot_labels.columns.str.contains(filter_criteria_goal[0])].columns)] = 0
        inverse_transition_label_tensor=torch.tensor(inverse_transition_labels.values).float()



        #can plot umap for diagnostic purposes, how well is the model performing at transitioning the cells to and from the goal?
        if umap:
            plt.close('all')
            fig, ((ax1, ax2,ax3,ax4)) = plt.subplots(nrows=1, ncols=4,figsize=(40,10))
            _=plot_umap({
                "Start":start_state_tensor.clone().detach().cpu().numpy(),
                "Goal":goal_state_tensor.clone().detach().cpu().numpy(),
                "S2G_transformed":stargan.G(start_state_tensor.to(device),transition_label_tensor.to(device)).clone().detach().cpu().numpy(),
                "G2S_transformed":stargan.G(goal_state_tensor.to(device),inverse_transition_label_tensor.to(device)).clone().detach().cpu().numpy()
            },n_neighbors=20, min_dist=0.5, sample_fraction=sample_fraction, point_size=5, alpha=alpha,title="UMAP of raw real data, stargan transitions",
                        ax=ax1
                       )
            
            torch.cuda.empty_cache()
            _=plot_umap({
                "Start_auto":stargan.G(start_state_tensor.to(device),start_label_tensor.to(device)).clone().detach().cpu().numpy(),
                "Goal_auto":stargan.G(goal_state_tensor.to(device),goal_label_tensor.to(device)).clone().detach().cpu().numpy(),
                "S2G_transformed":stargan.G(start_state_tensor.to(device),transition_label_tensor.to(device)).clone().detach().cpu().numpy(),
                "G2S_transformed":stargan.G(goal_state_tensor.to(device),inverse_transition_label_tensor.to(device)).clone().detach().cpu().numpy()
            },n_neighbors=20, min_dist=0.5, sample_fraction=sample_fraction, point_size=5, alpha=alpha,title="UMAP of stargan real data and transitions",
                        ax=ax2
                       )
            torch.cuda.empty_cache()
            _=plot_umap({
                "Start":start_state_tensor.clone().detach().cpu().numpy(),
                "Goal":goal_state_tensor.clone().detach().cpu().numpy(),
                "Goal_gen":stargan.G(goal_state_tensor.to(device),goal_label_tensor.to(device)).clone().detach().cpu().numpy(),
                "Start_gen":stargan.G(start_state_tensor.to(device),start_label_tensor.to(device)).clone().detach().cpu().numpy()
            },n_neighbors=20, min_dist=0.5, sample_fraction=sample_fraction, point_size=5, alpha=alpha,title="UMAP of raw real data, stargan initial state",
                        ax=ax3
                       )
            
            _=plot_umap({
                "Start":start_state_tensor.clone().detach().cpu().numpy(),  
                "Goal":goal_state_tensor.clone().detach().cpu().numpy(),
                "reconstructed_goal":stargan.G(stargan.G(goal_state_tensor.to(device),
                                                         inverse_transition_label_tensor.to(device)),
                                               goal_label_tensor.to(device)).clone().detach().cpu().numpy(),
                "reconstructed_start":stargan.G(stargan.G(start_state_tensor.to(device),
                                                          transition_label_tensor.to(device)),
                                                start_label_tensor.to(device)).clone().detach().cpu().numpy()
            },n_neighbors=20, min_dist=0.5, sample_fraction=sample_fraction, point_size=5, alpha=alpha,title="UMAP of raw real data, reconstructed data",
                        ax=ax4
                       )
            fig.savefig(result_path+"/"+identifier_label+'.png')

        
        torch.cuda.empty_cache()

        #diagnostic plot about performance. allows us to know if the oracle is fooled by the stargan
        if oracle_performance:
            real_start=oracle.D(start_state_tensor.to(device))
            fake_start=oracle.D(propagator.G(stargan.G(goal_state_tensor.to(device), torch.tensor(inverse_transition_labels.values).to(device))))
            real_goal=oracle.D(goal_state_tensor.to(device))
            fake_goal=oracle.D(propagator.G(stargan.G(start_state_tensor.to(device),torch.tensor(transition_labels.values).to(device))))
            real_start_accuracy ,real_start_distances=gosip.calculate_category_accuracy(predicted_logits=real_start,
                                    goal_categories=start_label_tensor.to(device),
                                       num_categories=num_categories)
            fake_goal_accuracy,fake_goal_distances=gosip.calculate_category_accuracy(predicted_logits=fake_goal,
                                    goal_categories=transition_label_tensor.to(device),
                                       num_categories=num_categories)
            fake_start_accuracy,fake_start_distances=gosip.calculate_category_accuracy(predicted_logits=fake_start,
                                    goal_categories=inverse_transition_label_tensor.to(device),
                                       num_categories=num_categories)
            real_goal_accuracy,real_goal_distances=gosip.calculate_category_accuracy(predicted_logits=real_goal,
                                    goal_categories=goal_label_tensor.to(device),
                                       num_categories=num_categories)
            gosip.plot_distances(category_labels, real_start_distances, fake_start_distances,
                           real_goal_distances, fake_goal_distances, result_path, "")
            





    
        
        numeric_dfs = []
        index_reference = None    
        gene_input_size = start_state_tensor.shape[1]  # or however many genes you test
        pval_matrix = np.zeros((k, gene_input_size))
        pval_matrix_cbnn = np.zeros((k, gene_input_size))
        # calculates oracle scores for each perturbation of each cell
        start_state_tensor=start_state_tensor.to(device)
        transition_label_tensor=transition_label_tensor.to(device)
        start_time = time.perf_counter()
        for i in range(k):
            with torch.inference_mode():
                gene_names=adata.var.index
                #violins_propagated, accuracy,change,oracle_tstat,oracle_pval,feature_importance,cbnn_results,cbnn_pval =full_perturbation_analysis_cbnn(start_tensor=start_state_tensor,
                #                                                                                    transition_label_tensor=transition_label_tensor,
                #                                                                                    stargan=stargan, oracle=oracle, propagator=propagator, 
                #                                                                                    num_categories=num_categories,cbnn=cbnn,
                #                                                                                    times=times,
                #                                                                                    propagation=True)
                violins_propagated, accuracy,change,oracle_tstat,oracle_pval,cbnn_results,cbnn_pval =full_perturbation_analysis_cbnn(start_tensor=start_state_tensor,
                                                                                    transition_label_tensor=transition_label_tensor,
                                                                                    stargan=stargan, oracle=oracle, propagator=propagator, 
                                                                                    num_categories=num_categories,cbnn=cbnn,
                                                                                    times=times,
                                                                                    propagation=True)
                perturbation_effect= np.median(violins_propagated, axis=1)
                            
                #print(f"number of zeros in network: {np.count_nonzero(feature_importance == 0)}")
                #prepare extra information for final results.
                perturbation_effect=pd.Series(perturbation_effect)
                perturbation_effect.index=adata.var.index
                perturbation_magnitude=pd.Series(change)
                perturbation_magnitude.index=adata.var.index
                torch.cuda.empty_cache()
                gc.collect()
                temp_scores=pd.DataFrame({'oracle_score': perturbation_effect,
                                         'suggested_perturbation': perturbation_magnitude,
                                         'label':adata.var.index.to_list()})
                

                median_survival_impact= np.median(cbnn_results, axis=1)
                temp_scores['survival_score']=median_survival_impact
                pval_matrix[i, :] = oracle_pval.copy()
                pval_matrix_cbnn[i, :] = cbnn_pval.copy()
                # Store label column from first iteration
                if i == 0:
                    label_col = temp_scores[['label']]
                    index_reference = temp_scores.index
                else:
                    # Ensure the same ordering and index across all runs
                    temp_scores = temp_scores.reindex(index_reference)
        
                numeric_dfs.append(temp_scores.drop(columns='label').astype(float))
                elapsed = time.perf_counter() - start_time
                avg_per_iter = elapsed / (i + 1)
                remaining_iters = k - (i + 1)
                eta = avg_per_iter * remaining_iters
            
                # print a one-line status
                clear_output(wait=True)
                print(f"Iter {i+1}/{k} done. — elapsed {elapsed:.1f}s — ETA {eta:.1f}s")
        # Stack all numeric DataFrames and compute mean across axis 0
        stacked = np.stack([df.values for df in numeric_dfs], axis=0)
        mean_values = stacked.mean(axis=0)
    
        avg_df = pd.DataFrame(mean_values, columns=numeric_dfs[0].columns, index=index_reference)
        avg_df['label'] = label_col
        # 1) only if k>1
        if k > 1:
            # sample std (ddof=1) across runs
            std_vals = stacked.std(axis=0, ddof=1)             # shape (n_rows, n_cols)
        
            # 2) standard error
            se_vals = std_vals / np.sqrt(k)                    # shape (n_rows, n_cols)
        
            # 3) build a DataFrame, add a suffix, and concat
            se_df = pd.DataFrame(
                se_vals,
                columns=numeric_dfs[0].columns,
                index=index_reference
            ).add_suffix('_se')                                 # e.g. 'pagerank' → 'pagerank_se'
        
            # now stick it on to your avg_df
            avg_df = pd.concat([avg_df, se_df], axis=1)
                
        centrality_scores=avg_df.copy()
        centrality_scores=centrality_scores.reindex(adata.var.index.to_list())
        
        
        
        def combine_pvalues_acat(pval_matrix: np.ndarray,
                                            apply_fdr: bool = False) -> np.ndarray:
            """
            Combine two-sided t-test p-values across k runs using ACAT with equal weights.
        
            Parameters
            ----------
            pval_matrix : np.ndarray, shape (k, p)
                Two-sided p-values from k repetitions for each of p tests.
            apply_fdr : bool, default=True
                If True, apply Benjamini–Hochberg FDR correction on the combined p-values.
        
            Returns
            -------
            combined_pvals : np.ndarray, shape (p,)
                The ACAT‐combined (and optionally FDR‐adjusted) p-values.
            """
            # 1) Clip to avoid infinities
            clipped = np.clip(pval_matrix, 1e-15, 1 - 1e-15)
        
            # 2) Transform to Cauchy variates: t_ij = tan[(0.5 - p_ij) * π]
            t = np.tan((0.5 - clipped) * np.pi)    # shape (k, p)
        
            # 3) Unweighted sum = mean across runs
            #    T_j = (1/k) * sum_i t_ij
            T = t.mean(axis=0)                     # shape (p,)
        
            # 4) Back‐transform to p‐values
            combined = 0.5 - np.arctan(T) / np.pi   # shape (p,)
        
            # 5) Optional BH‑FDR correction
            if apply_fdr:
                combined = multipletests(combined, method='fdr_bh')[1]
            return combined



        if k >1:
            centrality_scores["oracle_score_pval_acat"]= combine_pvalues_acat(pval_matrix, apply_fdr=False)
            centrality_scores["survival_score_pval_acat"]= combine_pvalues_acat(pval_matrix_cbnn, apply_fdr=False)
            centrality_scores["oracle_score_pval_acat_fdr"]= combine_pvalues_acat(pval_matrix, apply_fdr=True)
            centrality_scores["survival_score_pval_acat_fdr"]= combine_pvalues_acat(pval_matrix_cbnn, apply_fdr=True)
        else:
            centrality_scores["oracle_score_pval"]=oracle_pval.copy()
            centrality_scores["survival_score_pval"]=cbnn_pval.copy()
            centrality_scores["oracle_score_pval_fdr"]=multipletests(oracle_pval, method='fdr_bh')[1]
            centrality_scores["survival_score_pval_fdr"]=multipletests(cbnn_pval.copy(), method='fdr_bh')[1]
            
        #centrality_scores["directionToPreventDiabetes"]="norm"
        #helper column, to make it easier to know if the suggested perturbation is an activation or inhibition of expression.
        #centrality_scores.loc[centrality_scores["Difference_goal_minus_start"]>0,"directionToPreventDiabetes"]="ACTIVATING"
        #centrality_scores.loc[centrality_scores["Difference_goal_minus_start"]<0,"directionToPreventDiabetes"]="INHIBITORY"
        _=gosip.plot_top_features(violins=violins_propagated,names=gene_names,top_n=top_n,
                          title="Propagated genes that move "+filter_criteria_start[0]+ " to " + filter_criteria_goal[0]  + " the most: 90% C.I.",
                          location=result_path + "/Propagated_"+filter_criteria_start[0] + "2" + filter_criteria_goal[0] +".pdf")        

    centrality_scores.index.name="gene_name"
    centrality_scores['k']=k
    centrality_scores.to_csv(result_path+"summary_statistics.csv")
    return centrality_scores, accuracy, result_path

