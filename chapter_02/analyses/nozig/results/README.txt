
## README - Analysis Pipeline Output
This directory contains the results generated from the **complete_analysis_pipeline** function.
The outputs are saved in an **Excel file** (`*_analysis_results.xlsx`) with multiple sheets summarizing each analysis.

## Visualizations
- Scatterplot of gene categorization results by driver-forward tiers.
- Data distribution plots for HPAP and developmental datasets.
- Simulation result plots.

### Excel File: *_analysis_results.xlsx
The file includes the following sheets:
---
### 1. Start_to_Goal
- **Forward transition:** Start -> Goal, sorted by `oracle_score`.
- **Columns:**
  - `label` – Gene label.
  - `oracle_score` – Oracle score in [−1, 1].
  - `pval_acat_fdr` – BH-adjusted combined p-value.
  - `suggested_perturbation` – Δ(Goal − Start).
  - `out_degree_pagerank_positive` – Out‐degree PageRank centrality.
  - `in_degree_pagerank_positive` – In‐degree PageRank centrality.
  - (All other original result columns.)

### 2. Goal_to_Start
- **Reverse transition:** Goal -> Start, sorted by `oracle_score`.
- **Columns:** Same as the **1.** sheet.

### 3. activating_drugs
- Potential activating compounds mapped to forward-significant genes.
- Merged DrugBank (v5.1.13) with forward results.
- **Columns:**
  - `drugbank_id`
  - `drug_name`
  - `drug_description`
  - `effect_label` – Original activation/inhibition label.
  - `cellular_location_of_action`
  - `chromosomal_location_of_target`
  - `target_id`
  - `target_name`
  - `organism_tested`
  - `label` – Gene label.
  - `oracle_score`
  - `pval_acat_fdr`
  - `suggested_perturbation`
  - `out_degree_pagerank_positive`

### 4. inhibitory_drugs
- Potential inhibitory compounds similarly mapped.
- **Columns:** Same as **3.**

### 5. gene_categorization_results
- Integrated classification across both directions.
- **Columns:**
  - `oracle_score_Start_to_Goal`
  - `oracle_score_Goal_to_Start`
  - `pval_acat_fdr_Start_to_Goal`
  - `pval_acat_fdr_Goal_to_Start`
  - `pert_Start_to_Goal`
  - `pert_Goal_to_Start`
  - `out_pr_Start_to_Goal`
  - `out_pr_Goal_to_Start`
  - `in_pr_Start_to_Goal`
  - `in_pr_Goal_to_Start`
  - `p_value_category`
  - `regulator_category`
  - `driver_category`
  - `marker_category`
  - `driver_tier_forward`
  - `driver_tier_anti`
  - `marker_tier_forward`
  - `marker_tier_anti`

### 6. p_value_description
- Definitions for each `p_value_category`.

### 7. regulator_description
- Definitions for each `regulator_category`.

### 8. driver_forward_description
- Tier definitions for `driver_tier_forward`.

### 9. driver_anti_description
- Tier definitions for `driver_tier_anti`.

### 10. marker_forward_description
- Tier definitions for `marker_tier_forward`.

### 11. marker_anti_description
- Tier definitions for `marker_tier_anti`.

---

## Additional Information
- Inputs: Two CSVs (forward and reverse transitions) and a DrugBank CSV.
- Classification logic applies p-value thresholds, perturbation directionality, and PageRank quantiles.
- Description tables are included for transparency and reproducibility.
