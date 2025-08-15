
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from captum.attr import DeepLiftShap
from sklearn.model_selection import train_test_split
import time
from IPython.display import clear_output
import optuna
class PrepCbnnModel(nn.Module):
    def __init__(self, input_dim, layers,dropout=0.1):
        super(PrepCbnnModel, self).__init__()
        
        # Define the layers
        # Dynamically define the layers
        fc_layers = []
        
        # First layer
        fc_layers.append(nn.Linear(input_dim, layers[0]))
        fc_layers.append(nn.LeakyReLU())
        # fc_layers.append(nn.BatchNorm1d(layers[0])) 
        if dropout is not None:
            fc_layers.append(nn.Dropout(dropout))
        
        # Middle layers
        for i in range(len(layers) - 1):
            fc_layers.append(nn.Linear(layers[i], layers[i + 1]))
            fc_layers.append(nn.LeakyReLU())
            # fc_layers.append(nn.BatchNorm1d(layers[i + 1])) 
            if dropout is not None:
                fc_layers.append(nn.Dropout(dropout))
        
        # Final layer
        fc_layers.append(nn.Linear(layers[-1], 1))
        
        # Create the sequential model
        self.fc = nn.Sequential(*fc_layers)
        # Initialize weights using Xavier initialization
        self._initialize_weights()

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


def sample_case_base(data, time, event, ratio=10, comprisk=False, censored_indicator=None):
    def check_args_time_event(data, time, event):
        return {'time': time, 'event': event}

    def check_args_event_indicator(data, event_name, censored_indicator):
        event_numeric = pd.Categorical(data[event_name]).codes
        n_levels = len(np.unique(event_numeric))
        return {'event_numeric': event_numeric, 'n_levels': n_levels}

    var_names = check_args_time_event(data, time, event)
    time_var = var_names['time']
    event_name = var_names['event']

    modified_event = check_args_event_indicator(data, event_name, censored_indicator)
    event_var = modified_event['event_numeric']

    if not comprisk and modified_event['n_levels'] > 2:
        raise ValueError("For more than one event type, use comprisk=True or reformat data to a single event of interest.")

    survival_data = data[[time_var, event_name]].copy()

    n = len(survival_data)
    B = survival_data[time_var].sum()
    c = (survival_data[event_name] != 0).sum()
    b = ratio * c
    offset = np.log(B / b)

    prob_select = survival_data[time_var] / B
    which_pm = np.random.choice(n, size=b, replace=True, p=prob_select)
    b_series = survival_data.iloc[which_pm].copy()
    b_series[event_name] = 0
    b_series[time_var] = np.random.uniform(0, 1, b) * b_series[time_var]

    select_time_event = ~data.columns.isin([time_var, event_name])
    b_series = pd.concat([b_series.reset_index(drop=True), data.loc[which_pm, select_time_event].reset_index(drop=True)], axis=1)

    c_series = data[data[event_name] != 0]

    cb_series = pd.concat([c_series.reset_index(drop=True), b_series.reset_index(drop=True)])
    cb_series['offset'] = offset

    return cb_series





# Clear the output
def fit_hazard(cbnn_prep, epochs=20000, batch_size=500, val_data=None, patience=5,burn_in=5):
    # Check if GPU is available and the user wants to use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract offset and data, and move to device
    offset = torch.tensor(cbnn_prep['offset'].values, dtype=torch.float32).unsqueeze(1).to(device)  # Ensure offset is of shape [batch_size, 1]
    
    x_train = cbnn_prep['casebaseData'].drop(columns=cbnn_prep['eventVar'])
    x_train = x_train[cbnn_prep['features']]
    y_train = cbnn_prep['casebaseData'][cbnn_prep['eventVar']]
    x_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)  # Ensure y_train is of shape [batch_size, 1]
    
    # Prepare DataLoader
    dataset = torch.utils.data.TensorDataset(x_train, y_train, offset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define the model, loss function, and optimizer, and move model to device
    model = cbnn_prep['network'].to(device)
    criterion = nn.BCELoss()  
    optimizer = optim.AdamW(model.parameters())
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()  # Start time of the epoch
        model.train()
        for inputs, targets, offsets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, offset=offsets)
            loss = criterion(outputs, targets)  # Ensure dimensions match
            loss.backward()
            optimizer.step()

        
        
        # Validation (if val_data is provided)
        
        if val_data is not None:
            offset_val = torch.tensor(val_data['offset'].values, dtype=torch.float32).unsqueeze(1).to(device)  # Ensure offset_val is of shape [batch_size, 1]
            x_val = val_data.drop(columns=cbnn_prep['eventVar'])
            x_val = x_val[cbnn_prep['features']]
            y_val = val_data[cbnn_prep['eventVar']]
            x_val = torch.tensor(x_val.values, dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)  # Ensure y_val is of shape [batch_size, 1]
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val, offset=offset_val)
                val_loss = criterion(val_outputs, y_val)
                
                # Print epoch, validation loss, time taken, and patience counter
                end_time = time.time()  # End time of the epoch
                epoch_time = end_time - start_time
                clear_output(wait=True)
                print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss.item()}, Time: {epoch_time:.2f} seconds, Patience: {patience_counter}')
                if epoch >= burn_in:
                    if best_loss > val_loss.item():
                        best_loss = val_loss.item()
                        best_model_wts = model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                if patience_counter >= patience:
                    # Update cbnn_prep with results
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_wts)  # Load best weights
                    cbnn_prep['resultOfFit'] = {
                        'model': model,
                        'loss': loss.item()
                    }
                    cbnn_prep['xTrain'] = x_train
                    cbnn_prep['yTrain'] = y_train
                    return cbnn_prep
    
    # Update cbnn_prep with results
    model.load_state_dict(best_model_wts)  # Load best weights
    cbnn_prep['resultOfFit'] = {
        'model': model,
        'loss': loss.item()
    }
    cbnn_prep['xTrain'] = x_train
    cbnn_prep['yTrain'] = y_train
    return cbnn_prep

def split_data(df, test_size=0.2, val_size=0.25, random_state=None):
    """
    Splits a DataFrame into training, validation, and test sets.

    Parameters:
    - df (pd.DataFrame): The DataFrame to split.
    - test_size (float): Proportion of the dataset to include in the test set (between 0.0 and 1.0).
    - val_size (float): Proportion of the training+validation data to include in the validation set (between 0.0 and 1.0).
    - random_state (int, optional): Random seed for reproducibility.

    Returns:
    - train_data (pd.DataFrame): Training set.
    - val_data (pd.DataFrame): Validation set.
    - test_data (pd.DataFrame): Test set.
    """
    # Split the data into training+validation and test sets
    train_val_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

    # Split the training+validation data into training and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=val_size, random_state=random_state)
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    return train_data, val_data, test_data

# Normalizing function for convenience
def normalizer(data, means, sds, max_time):
    normalized = data.copy()
    for col in data.columns:
        if len(data[col].unique()) > 1:
            normalized[col] = (data[col] - means[col]) / sds[col]
    
    normalized['status'] = data['status']
    
    normalized['time'] = data['time'] / max_time

    return normalized.copy()


def prep_cbnn(features, data, offset=np.nan, time_var='', event_var='', ratio=100, comp_risk=False, optimizer=None, layers=[10, 10], dropout=None, lr=0.001,device='cpu'):
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
    
    # Create the model with updated input_dim
    model = PrepCbnnModel(input_dim, layers,dropout=dropout).to(device)  # Move model to GPU

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(),lr=lr)

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


def shap_cbnn(fit, times, train_val, x_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Ensure model and data are on the correct device
    fit['network'].eval()
    device = next(fit['network'].parameters()).device

    # Select features from x_test and convert to tensor
    x_test = x_test[fit['features']].values
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    to_use = fit['features']
    print("Features used:", to_use)

    # Prepare the inputs for SHAP explainer
    x_train_tensor = torch.tensor(train_val[to_use].values, dtype=torch.float32).to(device)

    # Determine the time column index (assuming time is the last column)
    time_column = x_train_tensor.shape[1] - 1

    # Initialize results matrix
    results = np.full((len(times), x_test.shape[1] ), np.nan)

    # Perform predictions
    for i in range(len(times)):
        # Update time column for both train and test tensors
        x_train_tensor[:, time_column] = times[i]
        x_test_tensor[:, time_column] = times[i]

        train_val_inputs = (x_train_tensor, torch.zeros((train_val.shape[0], 1), device=device))  # Excluding time_var
        test_data_inputs = (x_test_tensor, torch.zeros((x_test_tensor.shape[0], 1), device=device))

        with torch.no_grad():
            # Create a DeepLiftShap object
            deep_shap = DeepLiftShap(fit['network'])


            # Calculate SHAP values
            shap_values = deep_shap.attribute(test_data_inputs, baselines=train_val_inputs)

            # Store the mean SHAP values into results
            results[i, :] = shap_values[0].mean(dim=0).detach().cpu().numpy()



    return results


def cu_inc_cbnn(fit, times, x_test):
    # Ensure model and data are on the correct device
    fit['network'].eval()
    device = next(fit['network'].parameters()).device

    # Create offset matrix and convert to tensor
    temp_offset = torch.zeros((x_test.shape[0], 1), device=device)

    # Select features from x_test and convert to tensor
    x_test = x_test[fit['features']].values
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    # Determine the time column index
    time_column = x_test_tensor.shape[1] - 1  # Adjust if time is not the last column

    # Initialize results matrix
    results = np.full((len(times), x_test.shape[0] + 1), np.nan)
    results[:, 0] = times

    # Perform predictions
    for i in range(len(times)):
        # Update the time column in x_test_tensor
        x_test_tensor[:, time_column] = times[i]
        with torch.no_grad():
            fit['network'].eval()
            # Perform prediction
            predictions = fit['network'](x_test_tensor, temp_offset).cpu().numpy()
        
        results[i, 1:] = predictions.flatten()

    # Calculate delta time
    delta_t = np.diff(times)[0]  # Assuming uniform time intervals

    # Calculate cumulative incidence
    for i in range(1, results.shape[1]):
        hazard = results[:, i] / (1 - results[:, i])  # Hazard function
        cumulative_hazard = np.cumsum(hazard * delta_t)  # Cumulative hazard
        results[:, i] = 1 - np.exp(-cumulative_hazard)  # Cumulative incidence
    
    times= results[:,0]
    
    return results[:,1:].T,times.T
