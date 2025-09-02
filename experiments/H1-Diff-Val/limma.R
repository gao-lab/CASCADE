# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %%
suppressPackageStartupMessages({
  library(dplyr)
  library(limma)
  library(edgeR)
})

# %% [markdown]
# # Read data

# %%
genes <- read.table(file.path(
  "../../data/validation/H1-diff/bulkRNA/resources/human/GENCODE_v37/genes",
  "gencodeV37_geneid_60710_uniquename_59453.txt"
), header = TRUE)
head(genes)

# %%
counts <- read.table(
  "../../data/validation/H1-diff/bulkRNA/results/03_quant/s02_counts.txt",
  header = TRUE
) %>%
  merge(genes)
counts <- counts %>%
  select(-c(gene_id, length, gene_name)) %>%
  filter(gene_type == "protein_coding") %>%
  rename(!!!c(
    PAX6.1 = "X20250527.GTZ.1.1",
    hESC.2 = "X20250527.GTZ.1.2",
    hESC.3 = "X20250527.GTZ.1.3",
    iNPC.d7.2 = "X20250527.GTZ.1.4",
    iAmacrine.1 = "X20250527.GTZ.2.1",
    iNPC.d7.3 = "X20250527.GTZ.2.2",
    iNPC.d7.4 = "X20250529.GTZ.1.2",
    iNPC.d4.1 = "X20250602.GTZ.1.1",
    iNPC.d4.2 = "X20250602.GTZ.1.2",
    iAmacrine.2 = "X20250701.GTZ.1.1",
    iAmacrine.3 = "X20250701.GTZ.1.2",
    PAX6.2 = "X20250706.GTZ.1.1",
    PAX6.3 = "X20250706.GTZ.1.2",
    TFAP2A.1 = "X20250716.GTZ.1.1",
    TFAP2A.2 = "X20250716.GTZ.1.2",
    TFAP2A.3 = "X20250716.GTZ.1.3",
    TFAP2A.4 = "X20250716.GTZ.1.4",
  )) %>%
  select(
    gene_unique_name,
    hESC.2, hESC.3,
    iNPC.d4.1, iNPC.d4.2,
    iNPC.d7.2, iNPC.d7.3, iNPC.d7.4,
    PAX6.1, PAX6.2, PAX6.3,
    TFAP2A.1, TFAP2A.2, TFAP2A.3, TFAP2A.4,
    iAmacrine.1, iAmacrine.2, iAmacrine.3,
  )
rownames(counts) <- counts$gene_unique_name
counts$gene_unique_name <- NULL
head(counts)

# %%
condition <- c(
  hESC.2 = "hESC", hESC.3 = "hESC",
  iNPC.d4.1 = "iNPC.d4", iNPC.d4.2 = "iNPC.d4",
  iNPC.d7.2 = "iNPC.d7", iNPC.d7.3 = "iNPC.d7", iNPC.d7.4 = "iNPC.d7",
  PAX6.1 = "PAX6", PAX6.2 = "PAX6", PAX6.3 = "PAX6",
  TFAP2A.1 = "TFAP2A", TFAP2A.2 = "TFAP2A", TFAP2A.3 = "TFAP2A", TFAP2A.4 = "TFAP2A",
  iAmacrine.1 = "iAmacrine", iAmacrine.2 = "iAmacrine", iAmacrine.3 = "iAmacrine",
)
condition <- factor(condition, levels = c(
  "hESC", "iNPC.d4", "iNPC.d7",
  "PAX6", "TFAP2A", "iAmacrine",
))

# %% [markdown]
# # Limma

# %%
design <- model.matrix(~condition)
design

# %%
vfit <- voomLmFit(counts, design = design, sample.weights = TRUE, plot = TRUE)
vfit <- eBayes(vfit)

# %%
const <- makeContrasts(
  TFAP2A.vs.iNPC.d7 = conditionTFAP2A - conditioniNPC.d7,
  PAX6.vs.iNPC.d7 = conditionPAX6 - conditioniNPC.d7,
  iAmacrine.vs.iNPC.d7 = conditioniAmacrine - conditioniNPC.d7,
  levels = design
)
cfit <- contrasts.fit(vfit, const)
cfit <- eBayes(cfit)

# %%
coef <- topTable(cfit, coef = "TFAP2A.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.TFAP2A.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "PAX6.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.PAX6.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "iAmacrine.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.iAmacrine.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "CREB5.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.CREB5.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "FOXD3.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.FOXD3.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "SOX10.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.SOX10.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "iSchwann.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.iSchwann.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "HES1.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.HES1.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "iMuller.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.iMuller.vs.iNPC.d7.csv", quote = FALSE)
coef <- topTable(cfit, coef = "RARB.vs.iNPC.d7", number = Inf) %>% arrange(desc(logFC))
write.csv(coef, "Limma.RARB.vs.iNPC.d7.csv", quote = FALSE)
