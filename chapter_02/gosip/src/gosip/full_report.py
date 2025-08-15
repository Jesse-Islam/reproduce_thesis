import os
import gc
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from .adata_utils import process_adata_to_tensors
from .neural_network_utils import calculate_category_accuracy
from .perturbation_impact_network_utils import (
    calculate_separate_centrality_from_adj,
    #adjacency_matrix_to_edge_list,
    #calculate_separate_centrality_dataframe,
    #set_values_between_percentiles_to_zero,
    preprocess_to_range_with_percentile_threshold
)
from .perturbation_impact_utils import full_perturbation_analysis #perturbation, process_gene_impact
from .visualization_utils import plot_distances, plot_top_features, plot_umap, plot_volcano
#from .genetic_algorithm import plot_ga_performance, GeneticSubsetSelector
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from IPython.display import clear_output
#import torch.nn.functional as F
def full_report(adata,stargan,oracle,propagator,
                shared_filter,filter_criteria_start,filter_criteria_goal,
                main_path,outdir,
                top_n=20,percentile=90,alpha=0.5,sample_fraction=1,
                num_categories=None,category_labels=None,device=None,
                *,umap=False,oracle_performance=False,custom_path=None,
                lambda_param=0.0,
                k=1,apply_fdr=True):

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
        start_adata,start_state_tensor, start_label_tensor = process_adata_to_tensors(shared_adata,1, filter_criteria_start)
        goal_adata,goal_state_tensor, goal_label_tensor = process_adata_to_tensors(shared_adata,1, filter_criteria_goal)
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
            real_start_accuracy ,real_start_distances=calculate_category_accuracy(predicted_logits=real_start,
                                    goal_categories=start_label_tensor.to(device),
                                       num_categories=num_categories)
            fake_goal_accuracy,fake_goal_distances=calculate_category_accuracy(predicted_logits=fake_goal,
                                    goal_categories=transition_label_tensor.to(device),
                                       num_categories=num_categories)
            fake_start_accuracy,fake_start_distances=calculate_category_accuracy(predicted_logits=fake_start,
                                    goal_categories=inverse_transition_label_tensor.to(device),
                                       num_categories=num_categories)
            real_goal_accuracy,real_goal_distances=calculate_category_accuracy(predicted_logits=real_goal,
                                    goal_categories=goal_label_tensor.to(device),
                                       num_categories=num_categories)
            plot_distances(category_labels, real_start_distances, fake_start_distances,
                           real_goal_distances, fake_goal_distances, result_path, "")
            





    
        
        numeric_dfs = []
        index_reference = None    
        gene_input_size = start_state_tensor.shape[1]  # or however many genes you test
        pval_matrix = np.zeros((k, gene_input_size))
        # calculates oracle scores for each perturbation of each cell
        start_state_tensor=start_state_tensor.to(device)
        transition_label_tensor=transition_label_tensor.to(device)
        for i in range(k):
            with torch.inference_mode():
                print(f'k-th iteration: {i}')
                gene_names=adata.var.index
                violins_propagated, accuracy, change, oracle_tstat, oracle_pval, feature_importance=full_perturbation_analysis(
                    start_tensor=start_state_tensor,
                    transition_label_tensor=transition_label_tensor,
                    stargan=stargan,oracle=oracle,propagator=propagator,
                    num_categories=num_categories,
                    lambda_param= lambda_param,
                    propagation=True
                )
                perturbation_effect= np.median(violins_propagated, axis=1)
                print(f"number of zeros in network: {np.count_nonzero(feature_importance == 0)}")
                #prepare extra information for final results.
                perturbation_effect=pd.Series(perturbation_effect)
                perturbation_effect.index=adata.var.index
                perturbation_magnitude=pd.Series(change)
                perturbation_magnitude.index=adata.var.index
                torch.cuda.empty_cache()
                gc.collect()
                #calculate centrality scores for perturbation impact network.
                temp_scores = calculate_separate_centrality_from_adj(adj_matrix=feature_importance,
                                                       perturbation_effect  = perturbation_effect,
                                                       perturbations_change = perturbation_magnitude,
                                                       labels=adata.var.index.to_list())
                pval_matrix[i, :] = oracle_pval
                # Store label column from first iteration
                if i == 0:
                    label_col = temp_scores[['label']]
                    index_reference = temp_scores.index
                else:
                    # Ensure the same ordering and index across all runs
                    temp_scores = temp_scores.reindex(index_reference)
        
                numeric_dfs.append(temp_scores.drop(columns='label').astype(float))
                clear_output(wait=True)
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
                                            apply_fdr: bool = True) -> np.ndarray:
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


        if apply_fdr:
            if k >1:
                centrality_scores["oracle_score_pval_acat_fdr"]= combine_pvalues_acat(pval_matrix,apply_fdr=apply_fdr)
            else:
                centrality_scores["oracle_score_pval_fdr"]=multipletests(oracle_pval, method='fdr_bh')[1]
        else:
            if k >1:
                centrality_scores["oracle_score_pval_acat"]= combine_pvalues_acat(pval_matrix,apply_fdr=apply_fdr)
            else:
                centrality_scores["oracle_score_pval"]=oracle_pval
            
        #centrality_scores["directionToPreventDiabetes"]="norm"
        #helper column, to make it easier to know if the suggested perturbation is an activation or inhibition of expression.
        #centrality_scores.loc[centrality_scores["Difference_goal_minus_start"]>0,"directionToPreventDiabetes"]="ACTIVATING"
        #centrality_scores.loc[centrality_scores["Difference_goal_minus_start"]<0,"directionToPreventDiabetes"]="INHIBITORY"
        _=plot_top_features(violins=violins_propagated,names=gene_names,top_n=top_n,
                          title="Propagated genes that move "+filter_criteria_start[0]+ " to " + filter_criteria_goal[0]  + " the most: 90% C.I.",
                          location=result_path + "/Propagated_"+filter_criteria_start[0] + "2" + filter_criteria_goal[0] +".pdf")        

    centrality_scores.index.name="gene_name"
    centrality_scores['k']=k
    centrality_scores.to_csv(result_path+"summary_statistics.csv")
    return centrality_scores, accuracy, result_path





