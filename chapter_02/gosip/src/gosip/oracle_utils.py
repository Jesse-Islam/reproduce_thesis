import time

import optuna
import torch
from IPython.display import clear_output
from torch import nn, optim

from .neural_network_utils import (
    WeightedBCEWithLogitsLoss,
    WeightedCrossEntropyLoss,
    #WeightedL1Loss,
    calculate_category_loss,
)


class OracleModel(nn.Module):
    def __init__(self, input_dim, num_categories,layers=None,dropout=0.1):
        super().__init__()
        if layers is None:
            error_message= "expects list of 3 ints"
            raise ValueError(error_message)
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layers[0], layers[1]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(layers[1], layers[2]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.classifiers = nn.ModuleList([nn.Linear(layers[2], out_features) for out_features in num_categories])
    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = []
        for classifier in self.classifiers:
            output = classifier(features)
            # Apply softmax at the last dimension to calculate probabilities for each category within the label
            outputs.append(output)
        # Concatenate the softmax results along columns, each block corresponds to a label's categories
        return  torch.cat(outputs, dim=1)



class Oracle:
    def __init__(self, input_dim, num_domains, device,learning_rate=0.0001, layer_d=None,drpt_d=0.2):
        
        """
        Class dedicated to a basic MLP used as the oracle. 
        Parameters:
         - input_dim: how many features?
         - num_domains: the number of categories we are predicting over. 
         - device: gpu or cpu?
         - learning_rate: shared by both optimizers (0.0001)
         - layer_d:  the nodes on each layer of the generator. (None)
         - dropout_d: dropout_rater (0.2)
        """
        
        if layer_d is None:
            error_message= "expects list of 3 ints"
            raise ValueError(error_message)
        self.device = device
        self.input_dim=input_dim
        self.num_domains=num_domains
        self.D = OracleModel(input_dim, num_domains,layers=layer_d,dropout=drpt_d).to(device)
        self.d_optimizer = optim.AdamW(self.D.parameters(), lr=learning_rate) #can add weight decay for restricting weights
        self.bce_loss = WeightedBCEWithLogitsLoss()
        self.ce_loss = WeightedCrossEntropyLoss()
        #self.l1_loss = WeightedL1Loss()#1,1,10,5 is good, but forms some self clusters in activated fibroblast
        self.lambda_adv = 1
        self.lambda_cls = 1
        self.lambda_rec = 1
        self.lambda_iden= 1
        self.lambda_gp = 0
        self.n_critic = 1

    def reset_grad(self):
        """Reset the gradient buffers.""" #FROM STARGAN
        self.d_optimizer.zero_grad()

    def train(self, dataloader,val_loader, num_epochs, patience=3, burn_in=0,*,optuna_run=False,trial=None,verbose=False):
        # Setup for early stopping
        self.before_burn=True
        best_loss = float('inf')
        if not optuna_run:
            best_d_model_wts = self.D.state_dict().copy()
        epochs_without_improvement = 0
        start_time=time.time()
        for epoch in range(num_epochs):
            self.D.train()
            epoch_loss = 0
            i=0
            for data, labels,weights,global_weights in dataloader:
                start_data = data.to(self.device)
                start_labels = labels.to(self.device)
                weights= weights.to(self.device)
                global_start_weights=global_weights.to(self.device)
                # Discriminator training
                predicted_start_labels = self.D(start_data)
                d_loss_cls = calculate_category_loss(self.ce_loss,
                                    predicted_categories=predicted_start_labels,
                                    goal_categories=start_labels,
                                    num_categories=self.num_domains,
                                    weights_all=global_start_weights)

                d_loss = self.lambda_cls * (d_loss_cls)
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                epoch_loss=epoch_loss + d_loss.item()
                #if i==int(len(dataloader)/10):
                #    break
                i=i+1
            # Average loss for the epoch
            epoch_loss = (epoch_loss)/(len(dataloader)*data.shape[0])
            # Validation phase
            val_loss = 0
            self.reset_grad()
            with torch.no_grad():
                self.D.eval()
                i=0
                for data, labels,weights,global_weights in val_loader:
                    start_data = data.to(self.device)
                    start_labels = labels.to(self.device)
                    #weights= weights/weights.shape[0]
                    weights= weights.to(self.device)
                    global_start_weights=global_weights.to(self.device)
                    predicted_start_labels = self.D(start_data) #measure of realness for start data
                    d_loss_cls = calculate_category_loss(self.ce_loss,
                                        predicted_categories=predicted_start_labels,
                                        goal_categories=start_labels,
                                        num_categories=self.num_domains,
                                        weights_all=global_start_weights)
                    d_loss =  self.lambda_cls * (d_loss_cls)
                    val_loss=val_loss + d_loss.item()
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
                    best_d_model_wts = self.D.state_dict().copy()
                    self.before_burn=False
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                    if not optuna_run:
                        best_d_model_wts = self.D.state_dict().copy()
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Stopping early at epoch {epoch}")
                    clear_output(wait=True)
                    self.D.load_state_dict(best_d_model_wts)
                    return val_loss
            if verbose and (epoch+1) %1==0:
                clear_output(wait=True)
                print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f},  Seconds per epoch: {((time.time()-start_time)/(epoch+1)):.2f}, Without improvement: {epochs_without_improvement}')
        clear_output(wait=True)
        if not optuna_run:
            self.D.load_state_dict(best_d_model_wts)
        return val_loss



