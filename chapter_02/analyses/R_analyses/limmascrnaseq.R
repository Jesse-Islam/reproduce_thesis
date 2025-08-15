
main_path="/home/jislam/Documents/gits/reproduce_thesis/chapter_02/analyses/nozig/"
dataLocation="/home/jislam/Documents/gits/scRNAseq_datasets/hpap_filtered3k.h5ad"


library(zellkonverter)
library(SingleCellExperiment)
library(scuttle)            # aggregateAcrossCells
library(edgeR)
library(variancePartition)  # voomWithDreamWeights, dream
library(BiocParallel)


pseudobulk_dream <- function(sce, cluster1, cluster2) {

  
  # parallel backend
  param <- SnowParam(17, "SOCK", progressbar = TRUE)
  
  # 1) Pseudobulk by donor × cell_disease
  grouping <- colData(sce)[, c("donor_id", "cell_disease"), drop = FALSE]
  sce_pb   <- aggregateAcrossCells(sce, ids = grouping,use.assay.type = "counts")
  
  # 2) Assign column names from metadata
  pb_meta   <- as.data.frame(colData(sce_pb), stringsAsFactors = FALSE)
  new_names <- make.unique(paste(pb_meta$donor_id, pb_meta$cell_disease, sep = "_"))
  colnames(sce_pb) <- new_names
  rownames(pb_meta) <- new_names
  
  # 3) Extract counts and filter to clusters of interest
  expr <- assay(sce_pb, "counts")
  pb_meta$cell_disease <- make.names(pb_meta$cell_disease)
  keep <- pb_meta$cell_disease %in% c(cluster1, cluster2)
  expr <- expr[, keep, drop = FALSE]
  meta <- pb_meta[keep, , drop = FALSE]
  
  # 4) Prepare metadata for modeling
  meta$cell_disease <- factor(meta$cell_disease, levels = c(cluster1, cluster2))
  meta$IDs <- factor(meta$donor_id)
  
  # 5) Build DGEList and normalize
  dge <- DGEList(expr)
  dge <- calcNormFactors(dge)
  
  # 6) Set up model and contrast
  form <- ~ cell_disease - 1
  L <- makeContrastsDream(
    form, meta,
    contrasts = c(
      t2doverctrl = paste0(
        "cell_disease", cluster1,
        " - cell_disease", cluster2
      )
    )
  )
  
  # 7) Fit dream model and compute statistics
  vobj <- voomWithDreamWeights(dge, form, meta, BPPARAM = param)
  fit  <- dream(vobj, form, meta, BPPARAM = param, L = L)
  fit  <- eBayes(fit)
  
  # 8) Return full results table
  topTable(fit, coef = "t2doverctrl", number = Inf)
}

# Load h5ad file
sce <- readH5AD(dataLocation)


t2doverctrldelta <- pseudobulk_dream(
  sce,
  cluster1="delta.cell_T2D",
  cluster2="delta.cell_Control"
)
t2doverctrlbeta <- pseudobulk_dream(
  sce,
  cluster1="beta.cell_T2D",
  cluster2="beta.cell_Control"
)

t2doverctrlalpha <- pseudobulk_dream(
  sce,
  cluster1 = "alpha.cell_T2D",
  cluster2 = "alpha.cell_Control"
)



rowname_column_namer<-function(results){
df <- cbind(gene_name = row.names(results), results)
return(df)
}

write.csv(rowname_column_namer(t2doverctrldelta),file='t2doverctrldelta.csv', row.names = FALSE)
write.csv(rowname_column_namer(t2doverctrlalpha),file='t2doverctrlalpha.csv', row.names = FALSE)
write.csv(rowname_column_namer(t2doverctrlbeta),file='t2doverctrlbeta.csv', row.names = FALSE)




