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
  library(forcats)
  library(ggplot2)
  library(gtools)
  library(readr)
  library(reshape2)
  library(stringr)
  library(tidyr)
  library(yaml)
})

# %%
display <- read_yaml("../../config/display.yaml")
names(display$naming$datasets) <- gsub(
  "<br>", "\n", names(display$naming$datasets)
)

# %%
summary_all <- read_csv(
  "../../sum/actual_dsgn.csv",
  col_type = c(ds = "f", scf = "f", aux = "f", run = "f", phs = "f")
) %>%
  mutate(
    ds = ds %>%
      fct_recode(!!!display$naming$datasets) %>%
      fct_relevel(names(display$naming$datasets)),
    imp = factor(imp),
    n_vars = factor(n_vars),
    kg = factor(kg),
    kc = factor(kc),
    div_sd = factor(div_sd),
    meth = meth %>%
      {
        .[grepl("dsgn=sb$", run)] <- "cascade_sb"
        .
      } %>%
      {
        .[grepl("dsgn=bf$", run)] <- "cascade_bf"
        .
      } %>%
      as.factor() %>%
      fct_recode(!!!display$naming$methods) %>%
      fct_relevel(names(display$naming$methods))
  ) %>%
  sample_frac()
display$naming$metrics <- display$naming$metrics %>%
  {
    .[. %in% colnames(summary_all)]
  }
summary_all <- summary_all %>% rename(!!!display$naming$metrics)
str(summary_all)

# %% [markdown]
# # Exact match

# %% [markdown]
# ## Train

# %%
options(repr.plot.width = 7, repr.plot.height = 3.7)
gp <- ggplot(
  data = summary_all %>% filter(phs == "train"),
  mapping = aes(x = ds, y = AUHRC, col = meth)
) +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.6),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.6),
  ) +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(x = "Dataset", col = "Method") +
  guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
  ggplot_theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    legend.position = "bottom"
  )
ggplot_save("../../sum/actual_dsgn_train.pdf", gp, width = 7, height = 3.7)
gp

# %% [markdown]
# ## Test

# %%
options(repr.plot.width = 7, repr.plot.height = 3.7)
gp <- ggplot(
  data = summary_all %>% filter(phs == "test"),
  mapping = aes(x = ds, y = AUHRC, col = meth)
) +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.6),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.6),
  ) +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(x = "Dataset", col = "Method") +
  guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
  ggplot_theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    legend.position = "bottom"
  )
ggplot_save("../../sum/actual_dsgn_test.pdf", gp, width = 7, height = 3.7)
gp

# %% [markdown]
# # Partial match

# %%
double_ds <- c("Norman-2019")
summary <- summary_all %>% filter(ds %in% double_ds)
str(summary)

# %% [markdown]
# ## Train

# %%
options(repr.plot.width = 4, repr.plot.height = 3)
gp <- ggplot(
  data = summary %>% filter(phs == "train"),
  mapping = aes(x = ds, y = `AUHRC (partial match)`, col = meth)
) +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.6),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.6),
  ) +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(x = "Dataset", col = "Method") +
  ggplot_theme()
ggplot_save("../../sum/actual_dsgn_train_partial.pdf", gp, width = 4, height = 3)
gp

# %% [markdown]
# ## Test

# %%
options(repr.plot.width = 4, repr.plot.height = 3)
gp <- ggplot(
  data = summary %>% filter(phs == "test"),
  mapping = aes(x = ds, y = `AUHRC (partial match)`, col = meth)
) +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.6),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.6),
  ) +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(x = "Dataset", col = "Method") +
  ggplot_theme()
ggplot_save("../../sum/actual_dsgn_test_partial.pdf", gp, width = 4, height = 3)
gp
