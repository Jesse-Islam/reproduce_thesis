#import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd
import networkit as nk
from scipy import sparse
def preprocess_to_range_with_percentile_threshold(matrix, percentile=90, remove_loops=True):
    """
    scale edge_weights beteen -1,1. technically this is already done during the score 
    calculation, so it shouldn't change anything.
    Thresholding based on separate percentiles for positive and negative values.
    
    - matrix: 2D numpy array.
    - percentile: The percentile threshold for positive and negative values (default: 90).
    - remove_loops: Whether to remove self-loops (default: True). I decided to not remove self loops (diagonal)
        because it makes more sense to keep it (we are manually forcing the perturbed gene to maintain its value.)
    Returns:
    - under threshold values are set to 0. scaled weights are returned in a 2d numpy array.
    """
    m = matrix.copy()

    # Step 1: Separate positive and negative values
    positive_values = m[m > 0]
    negative_values = m[m < 0]

    # Step 2: Calculate the positive and negative percentiles
    positive_threshold = np.percentile(positive_values, percentile) if len(positive_values) > 0 else 0
    negative_threshold = np.percentile(negative_values, 100 - percentile) if len(negative_values) > 0 else 0

    # Step 3: Set values between the percentiles to zero
    m = np.where((m > negative_threshold) & (m < positive_threshold), 0, m)

    # Step 4: Scale positive values to [0, 1]
    if len(positive_values) > 0:
        positive_values = m[m > 0]
        min_pos, max_pos = positive_values.min(), positive_values.max()
        if max_pos != min_pos:  # Avoid division by zero
            m[m > 0] = (m[m > 0] - min_pos) / (max_pos - min_pos)  # Scale to [0, 1]
        else:
            m[m > 0] = 1  # If all positive values are equal, set them to 1

    # Step 5: Scale negative values to [-1, 0]
    if len(negative_values) > 0:
        negative_values = m[m < 0]
        min_neg, max_neg = negative_values.min(), negative_values.max()
        if max_neg != min_neg:  # Avoid division by zero
            m[m < 0] = (m[m < 0] - max_neg) / (min_neg - max_neg)  # Scale to [-1, 0]
        else:
            m[m < 0] = -1  # If all negative values are equal, set them to -1

    # Step 6: Remove self-loops if specified
    if remove_loops:
        np.fill_diagonal(m, 0)  # Remove self-loops
    else:
        np.fill_diagonal(m, 1)  # Optionally, you could set the diagonal to 1 if self-loops are allowed

    return m



def calculate_separate_centrality_from_adj(adj_matrix, perturbation_effect, perturbations_change, labels):
    n = adj_matrix.shape[0]

    # Separate positive and negative adjacency matrices
    adj_positive = adj_matrix.copy()
    adj_positive[adj_positive <= 0] = 0



    # Helper function to compute centralities
    def compute_pagerank(adj):
        coo = sparse.coo_matrix(adj, dtype=float)

        G_fwd = nk.GraphFromCoo(coo, n=n, weighted=True, directed=True)
        G_rev = nk.GraphFromCoo(coo.transpose(), n=n, weighted=True, directed=True)

        pr_in = np.array(nk.centrality.PageRank(G_fwd, normalized=True).run().ranking())
        pr_out = np.array(nk.centrality.PageRank(G_rev, normalized=True).run().ranking())

        pr_in = pr_in[np.argsort(pr_in[:, 0]), 1]
        pr_out = pr_out[np.argsort(pr_out[:, 0]), 1]

        return pr_in, pr_out

    # Compute centralities for positive edges
    pr_in_pos, pr_out_pos = compute_pagerank(adj_positive)
    df = pd.DataFrame({
        "in_degree_pagerank_positive": pr_in_pos,
        "out_degree_pagerank_positive": pr_out_pos,
    }, index=labels)
    #adj_negative = adj_matrix.copy()
    #adj_negative[adj_negative >= 0] = 0
    #adj_negative = np.abs(adj_negative)
    # Compute centralities for negative edges (if any exist)
    #if np.any(adj_negative):
    #    pr_in_neg, pr_out_neg = compute_pagerank(adj_negative)
    #    df["in_degree_pagerank_negative"] = pr_in_neg
    #    df["out_degree_pagerank_negative"] = pr_out_neg

    # Zero-handling (replace zeros with half the smallest non-zero value)
    if (df > 0).any().any():
        m = df[df > 0].min().min() / 2
    else:
        m = 1e-9
    df = df.replace(0, m)

    # Add perturbation data
    df["oracle_score"] = perturbation_effect
    df["suggested_perturbation"] = perturbations_change
    df["label"] = df.index

    return df



def adjacency_matrix_to_edge_list(adj_matrix, labels=None):
    """
    Convert a weighted adjacency matrix to a Pandas DataFrame representing the edge list.
    
    Parameters:
    adj_matrix (np.array): A 2D numpy array where adj_matrix[i, j] represents the weight of the edge from vertex i to vertex j.
    labels (list): Optional list of labels corresponding to the vertices.
    Returns:
    pd.DataFrame: A DataFrame with columns ['source', 'target', 'weight', 'source_label', 'target_label'].
    """
    # Ensure the adjacency matrix is a numpy array
    adj_matrix = np.array(adj_matrix)

    # Find indices of non-zero elements
    source_indices, target_indices = np.nonzero(adj_matrix)

    # Get corresponding weights
    weights = adj_matrix[source_indices, target_indices]

    # If labels are not provided, use indices as labels
    if labels is None:
        labels = list(range(adj_matrix.shape[0]))

    # Convert labels to a NumPy array for efficient indexing
    labels = np.array(labels, dtype=str)

    # Create a DataFrame with the edge list
    edge_df = pd.DataFrame({
        'source': source_indices,
        'target': target_indices,
        'weight': weights,
        'source_label': labels[source_indices],
        'target_label': labels[target_indices]
    })
    return edge_df



def calculate_separate_centrality_dataframe(edge_df, perturbation_effect, perturbations_change,labels):
    """
    takes a edge_list and returns a matrix with centrality measures of interest,
    while attaching the oracle_score and the scaled, suggested perturbation amount
    Parameters:
    - edge_df:  A DataFrame with columns ['source', 'target', 'weight', 'source_label', 'target_label']
    - perturbation_effect: a numpy array or pandas Series with the oracle_score.
    - perturbations_change: a numpy array or pandas Series with the suggested 
      perturbation amount for each perturbation.
    - labels: feature labels.
    Returns: dataframe with all results that were calculated and concatenated.
    
    """

    # Function to calculate centrality measures
    def calculate_centrality_measures(df, substring,labels):
        row = df['source'].values.astype(int)
        col = df['target'].values.astype(int)
        weights = df['weight'].values.astype(float)

        G = nk.GraphFromCoo((weights,(row,col)), n = len(labels),weighted=True,directed=True)
        pg=np.array(nk.centrality.PageRank(G,normalized=True).run().ranking())
        in_pagerank=pg[pg[:, 0].argsort()]
        #bt=np.array(nk.centrality.Betweenness(G,normalized=True).run().ranking())
        #betweenness_centrality=bt[bt[:, 0].argsort()]
        
        G = nk.GraphFromCoo((weights,(col,row)), n = len(labels),weighted=True,directed=True)
        pgr=np.array(nk.centrality.PageRank(G,normalized=True).run().ranking())
        out_pagerank=pgr[pgr[:, 0].argsort()]


        # Create DataFrame
        df = pd.DataFrame({
            f"in_degree_pagerank{substring}": in_pagerank[:,1],
            f"out_degree_pagerank{substring}": out_pagerank[:,1],
            #f"betweenness_centrality{substring}": betweenness_centrality[:,1]
        })
        df.index=[labels[i] for i in list(df.index)]
        return df

    # Separate positive and negative edges
    pos_edge_df = edge_df[edge_df['weight'] > 0]
    neg_edge_df = edge_df[edge_df['weight'] < 0]

    # Handle null graph cases
    if not pos_edge_df.empty:
        positive_centralities = calculate_centrality_measures(pos_edge_df, '_positive',labels=labels)
        #if not neg_edge_df.empty:
        #    negative_centralities = calculate_centrality_measures(neg_edge_df, '_negative',labels=labels)
        #    centralities = pd.concat([positive_centralities, negative_centralities], axis=1)
        #else:
        centralities = positive_centralities
        
        # Handle zeros
        if (centralities > 0).any().any():
            centralities[centralities == 0] = centralities[centralities > 0].min().min() / 2
        else:
            centralities[centralities == 0] = 1e-9

    # Add additional data
    centralities['oracle_score'] = perturbation_effect
    centralities['suggested_perturbation'] = perturbations_change
    centralities['label'] = centralities.index
    return centralities

