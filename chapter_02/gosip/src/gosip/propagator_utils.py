# code adapted from https://github.com/rtqichen/beta-tcvae
##THIS IS THE CUSTOM VERSION
import time
import math
import optuna

import torch
from torch import nn, optim
import torch.nn.functional as F

from IPython.display import clear_output

import torch
from torch.distributions import Normal

class VAE(nn.Module):
    """
    Variational Autoencoder with a Zero-Inflated Gaussian decoder.
    """
    def __init__(self, input_dim, layers=[100,50,100], latent_dim=10, dropout=0.1,zi=True):
        super(VAE, self).__init__()
        # Encoder
        self.zi=zi
        if self.zi:
            self.output_multiplier= 3
            
        else:
            self.output_multiplier= 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layers[0], layers[1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        # Latent stats
        self.mu_layer = nn.Linear(layers[1], latent_dim)
        self.logvar_layer = nn.Linear(layers[1], latent_dim)
        # Decoder outputs 3 or 2*input_dim: pi_logits, mu, logvar
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layers[2]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layers[2], self.output_multiplier * input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, reconstruct_only=True):
        """
        If reconstruct_only=True, returns a stochastic draw from the zero-inflated Gaussian.
        Else returns raw decoder params and latent stats for training.
        """
        # pre-define clamp ranges
        min_logit, max_logit = -5.0, 5.0
        min_s, max_s       = -5.0, 5.0
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        params = self.decoder(z)
        if reconstruct_only:
                if self.zi:
                    # zero-inflated: unpack 3 heads
                    pi_logit, mu, s_zig = params.chunk(3, dim=1)
            
                    # π = sigmoid(clamp(pi_logit)) clamped to [ε,1−ε]
                    pi_logit.clamp_(min_logit, max_logit)
                    pi = pi_logit.sigmoid_().clamp_(1e-6, 1 - 1e-6)
            
                    # σ = exp(clamp(s_zig))
                    s_zig.clamp_(min_s, max_s)
                    sigma = s_zig.exp_()
            
                    # sample Gaussian noise
                    eps = torch.randn_like(mu)
                    # x = μ + σ * ε  (in-place mul/add)
                    x = eps.mul_(sigma).add_(mu)
            
                    # sample mask via compare: mask=1 if keep, 0 if zero
                    # (1−π) chance of keep → torch.rand > π
                    mask = (torch.rand_like(pi) > pi).to(mu.dtype)
            
                    return x.mul_(mask)
                else:
                    # plain Gaussian: unpack 2 heads
                    mu, s_zig = params.chunk(2, dim=1)
            
                    # σ = exp(clamp(s_zig))
                    s_zig.clamp_(min_s, max_s)
                    sigma = s_zig.exp_()
            
                    # sample Gaussian noise and return
                    eps = torch.randn_like(mu)
                    return eps.mul_(sigma).add_(mu)
        return params, mu, logvar, z


import torch
import torch.nn.functional as F

def reconstruction_loss_zig(data, params, eps=1e-12):
    """
    Zero-inflated Gaussian negative log-likelihood:
      • zero entries   -> –log(sigmoid(pi_logits))
      • nonzero entries-> –log(1–sigmoid(pi_logits)) + Gaussian NLL
    Args:
      data   : tensor of shape (batch, features)
      params : tensor of shape (batch, 3*features), concatenation of
               [pi_logits | mean | log_variance] along dim=1
      eps    : small constant for numerical stability
    Returns:
      Tensor of shape (batch,) with per-sample NLL
    """
    # unpack the three heads
    pi_logits, mean, log_variance = params.chunk(3, dim=1)

    # compute variance > 0
    variance = torch.exp(log_variance) + eps

    # mask zeros
    zero_mask = data.eq(0)

    # 1) mixing-weight loss: 
    #    –[y*log(pi) + (1–y)*log(1–pi)] with y = 1 for zeros
    bce = F.binary_cross_entropy_with_logits(
        pi_logits, zero_mask.float(), reduction='none'
    )

    # 2) Gaussian NLL: –log Normal(data | mean, variance)
    gauss_nll = F.gaussian_nll_loss(
        mean, data, variance, full=True, reduction='none'
    )

    # 3) combine: if zero use only bce, else add Gaussian term
    per_feature = torch.where(
        zero_mask,
        bce,
        bce + gauss_nll
    )

    # 4) sum over features to get per-sample NLL
    return per_feature.sum(dim=1)

'''
def reconstruction_loss_zig(data, params, eps=1e-12):
    """
    Zero-inflated Gaussian NLL, simplified:
      • zeros get log π
      • non-zeros get log[(1−π)·N(data|μ,σ²)]
    Returns a (batch,) tensor of NLLs.
    """
    # unpack the three heads
    pi_logits, mu, logvar = params.chunk(3, dim=1)

    # π ∈ [eps,1−eps], 1−π ≥ eps
    pi           = torch.sigmoid(pi_logits).clamp(eps, 1 - eps)
    one_minus_pi = (1 - pi).clamp(min=eps)

    # variance σ² ≥ eps
    var = logvar.exp().clamp(min=eps)

    # Gaussian log-density
    gauss_lp = -0.5 * ((data - mu).pow(2).div(var) + torch.log(2 * math.pi * var))

    # zero vs nonzero log-probs
    zero_lp    = torch.log(pi)
    nonzero_lp = torch.log(one_minus_pi) + gauss_lp

    # create mask for exact zeros
    zero_mask = data == 0  # shape (batch, features), dtype=bool

    # allocate empty tensor for log-probs
    lp = torch.empty_like(zero_lp)

    # fill in per-branch
    lp[zero_mask]    = zero_lp[zero_mask]
    lp[~zero_mask]   = nonzero_lp[~zero_mask]
    nll = -lp.sum(dim=1)

    return nll

        
def reconstruction_loss_gau(data, params, eps=1e-6, min_logvar=-5.0, max_logvar=5.0):
    """
    Gaussian negative log-likelihood per sample.
    """
    mu, logvar = params.chunk(2, dim=1)
    # Clamp log-variance for numerical stability
    logvar = logvar.clamp(min_logvar, max_logvar)

    # Variance > eps
    sigma2 = logvar.exp().clamp(min=eps)
    gauss_logprob = -0.5 * ((data - mu)**2 / sigma2 + torch.log(2 * math.pi * sigma2))
    return -gauss_logprob.sum(dim=1)
'''
import torch
import torch.nn.functional as F
import math

def reconstruction_loss_gau(data, params,
                                    eps: float = 1e-6,
                                    min_logvar: float = -5.0,
                                    max_logvar: float = 5.0):
    """
    Drop-in replacement using torch.nn.functional.gaussian_nll_loss

    Returns a Tensor of shape [batch] (per-sample NLL).
    """
    # Split into mean and log-variance
    mu, logvar = params.chunk(2, dim=1)
    # Stabilize logvar
    logvar = logvar.clamp(min_logvar, max_logvar)
    # Variance, enforced ≥ eps
    var = logvar.exp().clamp(min=eps)

    # Compute element-wise NLL (includes the 0.5*log(2π) term via full=True)
    # This returns a tensor of shape [batch, D] when reduction='none'
    nll_elem = F.gaussian_nll_loss(
        input=mu,
        target=data,
        var=var,
        eps=eps,
        reduction='none',
        full=True
    )
    # Sum over dimensions to get per-sample loss
    return nll_elem.sum(dim=1)



class BtcvaeLoss:
    def __init__(self, n_data, alpha=1.0, beta=6.0, gamma=1.0, is_mss=True,zi=True):
        self.n_data = n_data
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.is_mss = is_mss
        self.zi = zi
    def __call__(self, data, recon_batch, latent_dist, latent_sample, kl_weight=1.0,weight=1):
        batch_size, latent_dim = latent_sample.shape
        if self.zi:
            # Reconstruction Loss
            rec_loss = reconstruction_loss_zig(data, recon_batch)
        else:
            rec_loss = reconstruction_loss_gau(data, recon_batch)
        # Log densities
        #print(latent_dist)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
            latent_sample=latent_sample, latent_dist=latent_dist, n_data=self.n_data, is_mss=self.is_mss
        )

        # Mutual Information Loss
        mi_loss = (log_q_zCx - log_qz)#.mean()
        #print((log_q_zCx - log_qz).shape)
        # Total Correlation Loss
        tc_loss = (log_qz - log_prod_qzi)#.mean()

        # Dimension-Wise KL Loss
        dw_kl_loss = (log_prod_qzi - log_pz)#.mean()
        #print(rec_loss.shape,mi_loss.shape,tc_loss.shape,dw_kl_loss.shape)
        # Total Loss with KL annealing
        kl_loss = (self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss)
        loss = (weight*(rec_loss + kl_loss * kl_weight)).mean()
        #shared_ratio = rec_loss/kl_loss
        #loss = rec_loss + kl_loss*kl_weight*kl_loss/shared_ratio
        #return loss #+ the components for logging
        return loss#, {
        #    "rec_loss": rec_loss.detach().mean().cpu().item(),
        #    "kl_loss": kl_loss.detach().mean().cpu().item(),
        #    "mi_loss": mi_loss.detach().mean().cpu().item(),
        #    "tc_loss": self.beta*tc_loss.detach().mean().cpu().item(),
        #    "dw_kl_loss": dw_kl_loss.detach().mean().cpu().item()
        #}

def compute_kl_weight(epoch, burn_in, schedule="linear"):
    
    if schedule == "linear":
        return min(1.0, epoch / burn_in)
    elif schedule == "sigmoid":
        return 1 / (1 + math.exp(-10 * (epoch / burn_in - 0.5)))
    else:
        raise ValueError("Unsupported KL annealing schedule")


def log_density_gaussian(x, mu, logvar):
    normalization = -0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu) ** 2 * inv_var)
    return log_density

# Helper Functions
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)
    zeros = torch.zeros_like(latent_sample)
    
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)
    
    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)
    if is_mss:
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = (torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)-math.log(batch_size*n_data))       
    log_prod_qzi = (torch.logsumexp(mat_log_qz, dim=1, keepdim=False)-math.log(batch_size*n_data)).sum(1)
    return log_pz, log_qz, log_prod_qzi, log_q_zCx





def matrix_log_density_gaussian(x, mu, logvar):
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)

def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()




class Propagator:
    def __init__(self, input_dim, num_domains, device,learning_rate=0.0001, layer_g=None,drpt_g=0.2,latent_dim=10, zi=True):

        """
        Class dedicated to a tabular version of beta-tcvae. 
        Parameters:
         - input_dim: how many features?
         - num_domains: the number of categories we are predicting over. 
         - device: gpu or cpu?
         - learning_rate: shared by both optimizers (0.0001)
         - layer_g:  the nodes on each layer of the generator. (None)
         - dropout_g: dropout_rate (0.2)
         - latent_dim: how small should the compression be? (10)
        """

        
        if layer_g is None:
            error_message= "expects list of 3 ints"
            raise ValueError(error_message)
        self.device = device
        self.input_dim=input_dim
        self.num_domains=num_domains
        self.zi=zi
        self.G = VAE(input_dim,layers=layer_g,dropout=drpt_g,latent_dim=latent_dim,zi=self.zi).to(device)
        self.g_optimizer = optim.AdamW(self.G.parameters(), lr=learning_rate) #can add weight decay for restricting weights
        self.lambda_iden= 1.0
    def reset_grad(self):
        """Reset the gradient buffers.""" #FROM STARGAN
        self.g_optimizer.zero_grad()
    def train(self, dataloader,val_loader, num_epochs, patience=3, burn_in=10,*,kl_schedule="sigmoid",optuna_run=False,trial=None,verbose=False, loss_fn=None):
        # Setup for early stopping
        self.before_burn=True
        best_loss = float('inf')
        if not optuna_run:
            best_g_model_wts = self.G.state_dict().copy()
        epochs_without_improvement = 0
        start_time=time.time()
        for epoch in range(num_epochs):
            rec_total = kl_total = mi_total = tc_total = dw_kl_total = 0
            self.G.train()
            epoch_loss = 0
            kl_weight = compute_kl_weight(epoch, burn_in, schedule=kl_schedule)
            i=0
            for data, labels,weights,global_weights in dataloader:
                start_data = data.to(self.device)
                #start_labels = labels.to(self.device)
                weights= weights.to(self.device)
                global_start_weights=global_weights.to(self.device)
                transition_data_generated, mu, logvar, z = self.G(start_data,reconstruct_only=False)
                latent_dist = (mu, logvar)
                self.g_optimizer.zero_grad()
                #print(mu)
                #print(latent_dist[1])
                #print(z)g_loss, loss_details 
                #loss_identity_s, loss_details  = loss_fn(start_data, transition_data_generated, latent_dist, latent_sample=z, kl_weight=kl_weight,weight=global_start_weights)
                loss_identity_s  = loss_fn(start_data, transition_data_generated, latent_dist, latent_sample=z, kl_weight=kl_weight,weight=global_start_weights)
                g_loss = loss_identity_s *self.lambda_iden
                self.reset_grad()
                
                g_loss.backward()
                self.g_optimizer.step()
                epoch_loss = epoch_loss+ g_loss.item()
                #rec_total += loss_details["rec_loss"]
                #kl_total += loss_details["kl_loss"]
                #mi_total += loss_details["mi_loss"]
                #tc_total += loss_details["tc_loss"]
                #dw_kl_total += loss_details["dw_kl_loss"]
                #if i==int(len(dataloader)/10):
                #    break
                i=i+1
            # Average loss for the epoch
            epoch_loss = (epoch_loss)/(len(dataloader)*data.shape[0])
            # Validation phase
            val_loss = 0
            rec_total_v = kl_total_v = mi_total_v = tc_total_v = dw_kl_total_v = 0
            self.reset_grad()
            with torch.no_grad():
                self.G.eval()
                for data, labels,_,global_weights in val_loader:
                    start_data = data.to(self.device)
                    #start_labels = labels.to(self.device)
                    #weights= weights/weights.shape[0]
                    #weights= weights.to(self.device)
                    global_start_weights=global_weights.to(self.device)
                    transition_data_generated, mu, logvar, z = self.G(start_data,reconstruct_only=False)
                    latent_dist = (mu, logvar)
                    #loss_identity_s, loss_details_v = loss_fn(start_data, transition_data_generated, latent_dist, z, kl_weight=1,weight=global_start_weights)
                    loss_identity_s = loss_fn(start_data, transition_data_generated, latent_dist, z, kl_weight=1,weight=global_start_weights)
                    g_loss = loss_identity_s *self.lambda_iden
                    val_loss = val_loss+ g_loss.item()
                    #rec_total_v += loss_details_v["rec_loss"]
                    #kl_total_v += loss_details_v["kl_loss"]
                    #mi_total_v += loss_details_v["mi_loss"]
                    #tc_total_v += loss_details_v["tc_loss"]
                    #dw_kl_total_v += loss_details_v["dw_kl_loss"]
            val_loss=val_loss/(len(val_loader)*data.shape[0])
            
            # Early stopping condition and burn-in time
            if optuna_run:
                trial.report(val_loss, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.TrialPruned()
            elif epoch >= burn_in:
                if self.before_burn:
                    best_loss = val_loss
                    best_g_model_wts = self.G.state_dict().copy()
                    self.before_burn=False
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                    if not optuna_run:
                        best_g_model_wts = self.G.state_dict().copy()
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Stopping early at epoch {epoch}")
                    clear_output(wait=True)
                    self.G.load_state_dict(best_g_model_wts)
                    return val_loss
            clear_output(wait=True)        
            if verbose and ((epoch + 1) % 1 == 0) and (epoch >0):
                val_size=len(val_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Training Loss: {epoch_loss:.2f}, Validation Loss: {val_loss:.2f}")
                #print(f" Training-Recon: {rec_total/i:.2f}, KL: {kl_total/i:.2f} = MI: {mi_total/i:.2f}, TC: {tc_total/i:.2f}, DW-KL: {dw_kl_total/i:.2f}")
                #print(f" Validation-Recon: {rec_total_v/val_size:.2f}, KL: {kl_total_v/val_size:.2f} = MI: {mi_total_v/val_size:.2f}, TC: {tc_total_v/val_size:.2f}, DW-KL: {dw_kl_total_v/val_size:.2f}")
                print(f"  Seconds/Epoch: {(time.time() - start_time)/(epoch+1):.2f}, No Improvement: {epochs_without_improvement}")
                

        clear_output(wait=True)
        if not optuna_run:
            self.G.load_state_dict(best_g_model_wts)
        return val_loss