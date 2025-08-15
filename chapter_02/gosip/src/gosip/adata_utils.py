import numpy as np
import pandas as pd
import torch


def filter_adata_by_labels(adata, selected_labels, column_name):
    """
    Filters an AnnData object to only keep cells that match the specified labels in a given column.
    Parameters:
    adata (anndata.AnnData): The AnnData object to filter.
    selected_labels (list): A list of labels to keep.
    column_name (str): The name of the column in adata.obs used for filtering.
    Returns:
    anndata.AnnData: A filtered AnnData object containing only cells with the specified labels.
    """
    if column_name not in adata.obs.columns:
        error_message=column_name+ 'DNE in adata.obs'
        raise ValueError(error_message)
    # Create a mask that is True for rows where the column value is in selected_labels
    mask = adata.obs[column_name].isin(selected_labels)
    return adata[mask]


def one_hot_encode_combinations(df, col_names):
    """
    Creates a one-hot-encoding for each categorical variable of interest.
    Parameters:
    - df: dataframe that contains categorical variables of interest.
    - col_names: columns we wish to one-hot-encode.
    Returns:
    - final_encoded_df:the one-hot-encoded categories.
    - num_categories: the number of categories in each categorical variable of interest.
    """
    # Check for missing columns first
    if not all(col in df.columns for col in col_names):
        missing_cols = ', '.join([col for col in col_names if col not in df.columns])
        error_message="Missing columns: " +missing_cols
        raise ValueError(error_message)

    # Container for the one-hot encoded DataFrames
    encoded_dfs = []
    # List to store the number of categories in each column
    num_categories = []

    # Encode each column separately
    for col in col_names:
        # Convert to string for categorical handling
        temp_df = df[[col]].astype(str)
        # Create a new DataFrame where the column value is prefixed with the column name
        temp_df[col] = temp_df[col].apply(lambda x: f"{col}={x}") #KEEP THIS OR FIGURE OUT ANOTHER??
        # Apply one-hot encoding
        encoded_df = pd.get_dummies(temp_df[col])
        encoded_dfs.append(encoded_df)
        # Append the number of categories encoded for this column
        num_categories.append(encoded_df.shape[1])

    # Concatenate all the one-hot encoded DataFrames along axis=1 (columns)
    final_encoded_df = pd.concat(encoded_dfs, axis=1)

    # Return both the encoded DataFrame and the list of category counts
    return final_encoded_df.astype(int), num_categories






def process_adata_to_tensors(adata, sample_fraction, filter_criteria):
    """
    Processes an AnnData object to select a random subset, filter it, and convert to tensors.

    Parameters:
    - adata (anndata.AnnData): The input AnnData object.
    - sample_fraction (float): Fraction of the data to sample randomly (0.1 for 10%).
    - filter_criteria (np.array): A boolean array to filter rows based on 'one_hot_labels'.

    Returns:
    - tuple: A tuple containing two torch tensors (data tensor, labels tensor).
    """
    filter_indices = np.ones(adata.shape[0], dtype=bool)
    for criteria in filter_criteria:
        filter_indices &= adata.one_hot_labels.iloc[:, adata.one_hot_labels.columns.str.contains(criteria)].any(axis=1)

    filtered_adata = adata[filter_indices].copy()
    filtered_adata.one_hot_labels = adata.one_hot_labels[filter_indices].copy()
    # Step 1: Select a random subset of rows
    num_samples = int(filtered_adata.shape[0] * sample_fraction)
    if sample_fraction > 1.0:
        indices = np.random.choice(filtered_adata.n_obs, num_samples, replace=False)
        sampled_adata = filtered_adata[indices].copy()
        sampled_adata.one_hot_labels = filtered_adata.one_hot_labels.iloc[indices, :].copy()
    else:
        sampled_adata = filtered_adata
    # Step 3: Convert the filtered data and labels into PyTorch tensors
    data_tensor = torch.tensor(sampled_adata.X, dtype=torch.float32)
    labels_tensor = torch.tensor(sampled_adata.one_hot_labels.values, dtype=torch.float32)

    return sampled_adata, data_tensor, labels_tensor

