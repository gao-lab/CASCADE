suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
  library(dplyr)
})


load("human_BM_Ery_smart_seq2.Rdata")

so <- CreateSeuratObject(
  Ery.integrated_counts_matrix,
  meta.data = Ery.integrated_meta_info %>% mutate(
    Plate_ID = as.character(Plate_ID),
    Sample_ID = as.character(Sample_ID),
    Phase = as.character(Phase),
    cluster = as.character(cluster)
  )
)
so[["umap"]] <- CreateDimReducObject(
  as.matrix(Ery.integrated.cell.embeddings),
  key = "UMAP_"
)

SaveH5Seurat(so, "human_BM_Ery_smart_seq2.h5Seurat", overwrite = TRUE)
Convert("human_BM_Ery_smart_seq2.h5Seurat", dest = "h5ad", overwrite = TRUE)
