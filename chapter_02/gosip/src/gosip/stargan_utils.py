import time
import math
import numpy as np
import optuna
import torch
from IPython.display import clear_output
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import geomloss


from .neural_network_utils import (
    WeightedBCEWithLogitsLoss,
    WeightedCrossEntropyLoss,
    WeightedL1Loss,
    calculate_category_loss,
)


import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class WeightedZIGNLLLoss(nn.Module):
    """
    Weighted Zero-Inflated Gaussian NLL Loss,
    mixing a point-mass at zero (pi) with a Gaussian slab elsewhere.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu: torch.Tensor,          # [..., D]
        sigma: torch.Tensor,       # same shape
        pi: torch.Tensor,          # same shape, probabilities in (0,1)
        x: torch.Tensor,           # same shape, observations
        weights: torch.Tensor,     # same shape (or broadcastable)
        min_sigma: float = 1e-3,
        max_sigma: float = 1e2,
    ) -> torch.Tensor:
        # 1) clamp pi into (eps, 1-eps)
        pi = pi.clamp(self.eps, 1 - self.eps)

        # 2) clamp sigma and form variance
        sigma = sigma.clamp(min_sigma, max_sigma)
        var   = sigma.pow(2)

        # 3) binary cross‐entropy for the zero vs nonzero choice
        #    target = 1 where x==0, else 0
        zero_mask = x.eq(0).float()
        mixing_nll = F.binary_cross_entropy(
            pi, zero_mask, reduction='none'
        )
        #    mixing_nll = -[y*log(pi) + (1-y)*log(1-pi)]

        # 4) Gaussian NLL for the continuous part
        #    full=True adds the 0.5*log(2*pi*var) term
        gaussian_nll = F.gaussian_nll_loss(
            mu, x, var, full=True, reduction='none'
        )
        #    gaussian_nll = -log Normal(x | mu, var)

        # 5) add gaussian term only for nonzeros
        nonzero_mask = 1.0 - zero_mask
        per_elem_nll = mixing_nll + nonzero_mask * gaussian_nll

        # 6) apply weights and average
        weighted_nll = per_elem_nll * weights
        return weighted_nll.mean()

'''
class WeightedZIGNLLLoss(nn.Module):
    """
    Weighted Zero-Inflated Gaussian NLL Loss,
    mixing a point-mass at zero (π) with a Gaussian slab elsewhere.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu: torch.Tensor,          # [..., 1] or [..., D]
        sigma: torch.Tensor,       # same shape
        pi: torch.Tensor,          # same shape
        x: torch.Tensor,           # same shape
        weights: torch.Tensor,     # same shape (or broadcastable)
        min_sigma: float = 1e-3,
        max_sigma: float = 1e2,
    ) -> torch.Tensor:
        # 1) clamp π into (eps,1−eps), compute 1−π
        pi = pi.clamp(self.eps, 1 - self.eps)
        one_minus_pi = (1 - pi)

        # 2) clamp σ into [min_sigma, max_sigma], get var
        sigma = sigma.clamp(min_sigma, max_sigma)
        var   = sigma.pow(2)

        # 3) Gaussian log‐density at x
        log_gauss = -0.5 * (
            torch.log(2 * math.pi * var)
            + (x - mu).pow(2).div(var)
        )

        # 4) masks for exact zero vs nonzero
        zero_mask    = (x == 0).float()
        nonzero_mask = 1.0 - zero_mask

        # 5) log‐probabilities of each branch
        log_prob_zero    = torch.log(pi)
        log_prob_nonzero = torch.log(one_minus_pi) + log_gauss

        # 6) select per‐element log‐prob
        log_prob = zero_mask * log_prob_zero + nonzero_mask * log_prob_nonzero

        # 7) negative log‐likelihood, apply weights, and average
        nll = -log_prob
        weighted_nll = nll * weights
        return weighted_nll.mean()

class WeightedGNLLLoss(nn.Module):
    """
    Weighted Gaussian Negative Log-Likelihood Loss.
    Expects inputs:
      mu:    (batch, features) mean of Gaussian
      sigma: (batch, features) scale of Gaussian (>0)
      pi:    (batch, features) zero-inflation prob in (0,1)
      x:     (batch, features) observed targets
      weights: (batch, features) elementwise weights
    Returns:
      scalar loss = mean(weighted nll)
    """
    def __init__(
        self,
        eps: float = 1e-6,
        min_sigma: float = 1e-3,
        max_sigma: float = 1e2,
        max_var: float   = 1e4
    ):
        super().__init__()
        self.eps       = eps
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_var   = max_var
    def forward(self, mu, sigma, x, weights):
        # variance
        sigma = sigma.clamp(min_sigma, max_sigma)
        var   = sigma.pow(2).clamp(min=self.eps)
        # log density of Gaussian
        log_gauss = -0.5 * (torch.log(2 * math.pi * var) + (x - mu)**2 / var)

        nll = -log_gauss
        # apply weights
        weighted = nll * weights
        return weighted.mean()
'''


class WeightedGNLLLoss(nn.Module):
    """
    Weighted Gaussian Negative Log-Likelihood Loss using PyTorch's built-in.
    Expects inputs:
      mu:      (batch, features) predicted mean
      sigma:   (batch, features) predicted scale (>0)
      x:       (batch, features) targets
      weights: (batch, features) elementwise weights
    Returns:
      scalar loss = mean(weighted NLL)
    """
    def __init__(
        self,
        eps: float = 1e-6,
        min_sigma: float = 1e-3,
        max_sigma: float = 1e2,
        max_var: float = 1e4
    ):
        super().__init__()
        self.eps       = eps
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_var   = max_var

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor,
                x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # 1) Clamp sigma and build variance
        sigma = sigma.clamp(self.min_sigma, self.max_sigma)
        var   = sigma.pow(2).clamp(min=self.eps, max=self.max_var)

        # 2) Element-wise Gaussian NLL (shape [batch, features])
        #    full=True adds the 0.5*log(2π) constant; use full=False to omit it
        nll_elem = F.gaussian_nll_loss(
            input=mu,
            target=x,
            var=var,
            eps=self.eps,
            reduction='none',
            full=True
        )

        # 3) Apply your weights, then do one final reduction
        weighted_nll = nll_elem * weights
        return weighted_nll.mean()


class Generator(nn.Module):
    def __init__(self, input_dim, num_domains, layers=None, dropout=0.1,
                 min_log_sigma=-10.0, max_log_sigma=10.0,
                 min_logit_pi=-5.0,  max_logit_pi=5.0,zi=True):
        super().__init__()
        if layers is None:
            raise ValueError("Generator expects list of 3 ints for layers")
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.min_logit_pi  = min_logit_pi
        self.max_logit_pi  = max_logit_pi
        self.zi=zi
        if self.zi:
            self.output_multiplier= 3    
        else:
            self.output_multiplier= 2
        self.network = nn.Sequential(
            nn.Linear(input_dim + np.sum(num_domains), layers[0]),
            nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(layers[0], layers[1]),
            nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(layers[1], layers[2]),
            nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(layers[2], input_dim * self.output_multiplier),
        )

    def forward(self, x, c, generate=True):
        # pack inputs and run through network
        h = torch.cat([x, c], dim=1)
        if  self.zi:
            mu, s, logit_pi = self.network(h).chunk(3, dim=1)

            # clamp the unconstrained outputs
            s        = s.clamp(self.min_log_sigma, self.max_log_sigma)
            logit_pi = logit_pi.clamp(self.min_logit_pi,  self.max_logit_pi)

            # parameterize strictly positive scale & valid probs
            sigma = torch.exp(s)               # σ = exp(s) ∈ (0,∞)
            pi    = torch.sigmoid(logit_pi)    # π = sigmoid(logit_pi) ∈ (0,1)
            if generate:
                # reparameterized Gaussian + Bernoulli mask
                eps     = torch.randn_like(mu)
                x_gauss = mu + sigma * eps
                mask    = torch.bernoulli(pi)      # 1⇒zero, 0⇒keep
                return x_gauss * (1 - mask)
    
            # for training/loss: return the params
            return mu, sigma, pi
        else:
            # pack inputs and run through network
            h = torch.cat([x, c], dim=1)
            mu, s= self.network(h).chunk(2, dim=1)
    
            # clamp the unconstrained outputs
            s        = s.clamp(self.min_log_sigma, self.max_log_sigma)
            
    
            # parameterize strictly positive scale & valid probs
            sigma = torch.exp(s)               # σ = exp(s) ∈ (0,∞)
            
    
            if generate:
                # reparameterized Gaussian + Bernoulli mask
                eps     = torch.randn_like(mu)
                x_gauss = mu + sigma * eps
                return x_gauss 
    
            # for training/loss: return the params
            return mu, sigma





class Discriminator(nn.Module):
    # unchanged from your original code
    def __init__(self, input_dim, num_categories, layers=None, dropout=0.1):
        super().__init__()
        if layers is None:
            raise ValueError("Discriminator expects list of 3 ints for layers")
        self.feature_extractor = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, layers[0])), nn.LeakyReLU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(layers[0], layers[1])), nn.LeakyReLU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(layers[1], layers[2])), nn.LeakyReLU(), nn.Dropout(dropout),
        )
        self.real_or_fake = spectral_norm(nn.Linear(layers[2], 1))
        self.classifiers = nn.ModuleList([
            spectral_norm(nn.Linear(layers[2], out_features)) for out_features in num_categories
        ])

    def forward(self, x):
        feats = self.feature_extractor(x)
        src = self.real_or_fake(feats)
        outs = [cls(feats) for cls in self.classifiers]
        return src, torch.cat(outs, dim=1)



class StarGAN:
    def __init__(
        self, input_dim, num_domains, device,
        learning_rate=1e-4, layer_g=[500,500,500], layer_d=[500,500,500],
        lambda_adv=1, lambda_cls=1, lambda_rec=10,# lambda_iden=10,
        critics=5, dropout_rate=0.1,zi=True
    ):
        # ... (keep your existing init, replacing L1 with ZIG)
        self.device = device
        self.zi=zi
        self.G = Generator(input_dim, num_domains, layers=layer_g, dropout=dropout_rate,zi=self.zi).to(device)
        self.D = Discriminator(input_dim, num_domains, layers=layer_d, dropout=dropout_rate).to(device)
        self.g_optimizer = optim.AdamW(self.G.parameters(), lr=learning_rate)
        self.d_optimizer = optim.AdamW(self.D.parameters(), lr=learning_rate)
        self.num_domains=num_domains
        self.bce_loss = WeightedBCEWithLogitsLoss()
        self.ce_loss = WeightedCrossEntropyLoss()
        
        if self.zi:
            self.rec_loss = WeightedZIGNLLLoss()
        else:
            self.rec_loss = WeightedGNLLLoss()
        self.lambda_adv = lambda_adv
        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        #self.lambda_iden = lambda_iden
        self.n_critic = critics
        self.dropout_rate = dropout_rate
        
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
    def _sample_fake(self, data, labels, generate=False):
        if self.zi:
            # zero-inflated path
            mu, sigma, pi = self.G(data, labels, generate=generate)
            # fast Bernoulli sampling: mask==1 keeps value, 0 zeroes it
            mask = (torch.rand_like(pi) > pi).to(mu.dtype)
            # fast Gaussian sampling via reparam trick
            eps = torch.randn_like(mu)
            noise = mu + sigma * eps
            fake = mask * noise
        else:
            # plain Gaussian path
            mu, sigma = self.G(data, labels, generate=generate)
            # fast Gaussian sampling via reparam trick
            eps = torch.randn_like(mu)
            fake = mu + sigma * eps
    
        return fake
    def _reconstruction_loss_calculation(self, fake, labels, target, avg_w):
        """
        Runs G on (fake, labels) and computes either:
          – zig_loss(mu, sigma, target, avg_w) if zi=False
          – rec_loss(mu, sigma, pi, target, avg_w)  if zi=True (zero-inflated)
        """
        if self.zi:
            # zero-inflated path: G returns (mu, sigma, pi)
            mu_rec, sigma_rec, pi_rec = self.G(fake, labels, generate=False)
            return self.rec_loss(mu_rec, sigma_rec, pi_rec, target, avg_w)
        else:
            # plain Gaussian path: G returns (mu, sigma)
            mu_rec, sigma_rec = self.G(fake, labels, generate=False)
            return self.rec_loss(mu_rec, sigma_rec, target, avg_w)

    def train(self, dataloader, val_loader, num_epochs, patience=3, burn_in=0,
              *, optuna_run=False, trial=None, verbose=False):
        # Setup for early stopping
        self.before_burn = True
        best_loss = float('inf')
        best_g_model_wts = self.G.state_dict().copy()
        best_d_model_wts = self.D.state_dict().copy()
        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.G.train(); self.D.train()
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0

            # ——— Training loop ———
            for data, labels, _, global_weights in dataloader:
                start_data   = data.to(self.device)
                start_labels = labels.to(self.device)
                global_w0    = global_weights.to(self.device)

                # ----- Discriminator updates -----
                for _ in range(self.n_critic):
                    # shuffle to get target domain
                    perm = torch.randperm(start_labels.size(0), device=self.device)
                    goal_data   = start_data[perm]
                    goal_labels = start_labels[perm]
                    global_w1   = global_w0[perm]
                    avg_w       = (global_w0 + global_w1) / 2.0

                    # 1) real
                    real_src, real_cls = self.D(start_data)
                    d_loss_real = self.bce_loss(real_src,
                                               torch.ones_like(real_src),
                                               global_w0)
                    d_loss_cls  = calculate_category_loss(
                        self.ce_loss,
                        predicted_categories=real_cls,
                        goal_categories=start_labels,
                        num_categories=self.num_domains,
                        weights_all=global_w0,
                    )

                    # 2) fake sg
                    fake_sg=self._sample_fake( start_data, goal_labels, generate=False)
                    # 3) fake gs
                    #fake_gs=self._sample_fake( goal_data, start_labels, generate=False)
                    
                    # adversarial loss
                    src_sg, _ = self.D(fake_sg.detach())
                    #src_gs, _ = self.D(fake_gs.detach())
                    d_loss_fake = ( self.bce_loss(src_sg,
                                                  torch.zeros_like(src_sg),
                                                  avg_w)
                                  #+ self.bce_loss(src_gs,
                                  #                torch.zeros_like(src_gs),
                                  #                avg_w)
                                  )# / 2.0

                    d_loss = (self.lambda_adv * (d_loss_real + d_loss_fake)
                              + self.lambda_cls * d_loss_cls)

                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
                    epoch_loss_d += d_loss.item()

                # ----- Generator update -----
                # re-use fake_sg and fake_gs from above
                src_sg, cls_sg = self.D(fake_sg)
                g_loss_adv_sg = self.bce_loss(src_sg,
                                              torch.ones_like(src_sg),
                                              avg_w)
                g_loss_cls_sg = calculate_category_loss(
                    self.ce_loss,
                    predicted_categories=cls_sg,
                    goal_categories=goal_labels,
                    num_categories=self.num_domains,
                    weights_all=avg_w,
                )

                g_loss_rec_sg = self._reconstruction_loss_calculation(fake_sg, start_labels, start_data, avg_w)
                
                #src_gs, cls_gs = self.D(fake_gs)
                #g_loss_adv_gs = self.bce_loss(src_gs,
                #                              torch.ones_like(src_gs),
                #                              avg_w)
                #g_loss_cls_gs = calculate_category_loss(
                #    self.ce_loss,
                #    predicted_categories=cls_gs,
                ##    goal_categories=start_labels,
                #    num_categories=self.num_domains,
                #    weights_all=avg_w,
                #)
                #g_loss_rec_gs = self._reconstruction_loss_calculation(fake_gs, goal_labels, goal_data, avg_w)

                # total generator loss
                g_loss = (
                    self.lambda_adv * (g_loss_adv_sg)# + g_loss_adv_gs)
                  + self.lambda_cls * (g_loss_cls_sg)# + g_loss_cls_gs)
                  + self.lambda_rec * (g_loss_rec_sg)# + g_loss_rec_gs)
                )# / 2.0

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                epoch_loss_g += g_loss.item()

            epoch_loss = (epoch_loss_g + epoch_loss_d) / (len(dataloader) * data.shape[0])

            # ——— Validation loop ———
            val_loss = 0.0
            self.reset_grad()
            with torch.no_grad():
                self.G.eval(); self.D.eval()
                for data, labels, weights, global_weights in val_loader:
                    start_data   = data.to(self.device)
                    start_labels = labels.to(self.device)
                    w0           = global_weights.to(self.device)

                    perm = torch.randperm(start_labels.size(0), device=self.device)
                    goal_data   = start_data[perm]
                    goal_labels = start_labels[perm]
                    w1          = w0[perm]
                    avg_w       = (w0 + w1) / 2.0

                    # sample fakes
                    fake_sg = self.G(start_data, goal_labels,generate=True)
                    #fake_gs = self.G(goal_data, start_labels,generate=True)

                    # recon losses
                    loss_sgs  = self._reconstruction_loss_calculation(fake_sg, start_labels, start_data, avg_w)
                    #loss_gsg  = self._reconstruction_loss_calculation(fake_gs, goal_labels, goal_data, avg_w)

                    

                    val_loss += ( self.lambda_rec * ( loss_sgs)) #/ 2.0 )loss_gsg +
                                #+ self.lambda_iden * loss_idv )

                    # classification on real
                    real_src, real_cls = self.D(start_data)
                    d_loss_cls = calculate_category_loss(
                        self.ce_loss,
                        predicted_categories=real_cls,
                        goal_categories=start_labels,
                        num_categories=self.num_domains,
                        weights_all=w0,
                    )
                    val_loss += self.lambda_cls * d_loss_cls

                val_loss = val_loss / (len(val_loader) * data.shape[0])

            # ——— Early Stopping & Optuna ———
            if optuna_run:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if epoch >= burn_in:
                if self.before_burn:
                    best_loss = val_loss
                    best_g_model_wts = self.G.state_dict().copy()
                    best_d_model_wts = self.D.state_dict().copy()
                    self.before_burn = False

                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                    if not optuna_run:
                        best_g_model_wts = self.G.state_dict().copy()
                        best_d_model_wts = self.D.state_dict().copy()
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Stopping early at epoch {epoch}")
                    clear_output(wait=True)
                    self.G.load_state_dict(best_g_model_wts)
                    self.D.load_state_dict(best_d_model_wts)
                    return val_loss

            if verbose and (epoch + 1) % 1 == 0:
                clear_output(wait=True)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {epoch_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Sec/epoch: {((time.time()-start_time)/(epoch+1)):.2f}, "
                    f"No Improve: {epochs_without_improvement}"
                )

        # end of epochs
        clear_output(wait=True)
        self.G.load_state_dict(best_g_model_wts)
        self.D.load_state_dict(best_d_model_wts)
        return best_loss