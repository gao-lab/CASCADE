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
  "../../sum/actual_dsgn_hparams.csv",
  col_type = c(ds = "f", sps = "f", acyc = "f", lik = "f", phs = "f")
) %>%
  select(-starts_with("edist"), -c(dat, scf, aux)) %>%
  mutate(
    ds = ds %>%
      fct_recode(!!!display$naming$datasets) %>%
      fct_relevel(names(display$naming$datasets)),
    n_vars = factor(n_vars),
    kg = factor(kg),
    kc = factor(kc),
    div_sd = factor(div_sd),
    nptc = factor(nptc),
    du = factor(du),
    dv = factor(dv),
    drop = factor(drop),
    lam = factor(lam),
    dec = factor(dec),
    run_sd = factor(run_sd),
  ) %>%
  sample_frac()

# %%
str(summary_all)

# %% [markdown]
# # Single

# %%
single_ds <- c(
  "Adamson-2016", "Replogle-2022\nRPE1",
  "Replogle-2022\nK562-ess", "Replogle-2022\nK562-gwps"
)

# %% [markdown]
# ## Train

# %%
summary <- summary_all %>%
  filter(ds %in% single_ds, phs == "train") %>%
  melt(variable.name = "metric_category", value.name = "value") %>%
  drop_na(value) %>%
  mutate(
    metric = str_match(metric_category, "^(.*) \\(")[, 2] %>%
      as.factor() %>%
      (function(x) fct_relevel(x, mixedsort(levels(x))))(),
    category = str_match(metric_category, "\\((.*)\\)$")[, 2]
  )
str(summary)

# %%
options(repr.plot.width = 6, repr.plot.height = 3.5)
gp <- ggplot(
  data = summary,
  mapping = aes(x = ds, y = value, col = lik)
) +
  facet_grid(
    metric_category ~ dsgn_sb,
    scales = "free_y",
    labeller = labeller(
      metric_category = function(x) paste("Metric:", x),
      dsgn_sb = function(x) paste("Optimize scale/bias:", x)
    )
  ) +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.5),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.5),
  ) +
  labs(x = "Dataset", y = "Metric value") +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_dsgn_train_single.pdf", gp,
  width = 6, height = 3.5
)
gp

# %% [markdown]
# ## Test

# %%
summary <- summary_all %>%
  filter(ds %in% single_ds, phs == "test") %>%
  melt(variable.name = "metric_category", value.name = "value") %>%
  drop_na(value) %>%
  mutate(
    metric = str_match(metric_category, "^(.*) \\(")[, 2] %>%
      as.factor() %>%
      (function(x) fct_relevel(x, mixedsort(levels(x))))(),
    category = str_match(metric_category, "\\((.*)\\)$")[, 2]
  )
str(summary)

# %%
options(repr.plot.width = 6, repr.plot.height = 3.5)
gp <- ggplot(
  data = summary,
  mapping = aes(x = ds, y = value, col = lik)
) +
  facet_grid(
    metric_category ~ dsgn_sb,
    scales = "free_y",
    labeller = labeller(
      metric_category = function(x) paste("Metric:", x),
      dsgn_sb = function(x) paste("Optimize scale/bias:", x)
    )
  ) +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.5),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.5),
  ) +
  labs(x = "Dataset", y = "Metric value") +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_dsgn_test_single.pdf", gp,
  width = 6, height = 3.5
)
gp

# %% [markdown]
# # Double

# %%
double_ds <- c("Norman-2019")

# %% [markdown]
# ## Train

# %%
summary <- summary_all %>%
  filter(ds %in% double_ds, phs == "train") %>%
  melt(variable.name = "metric_category", value.name = "value") %>%
  drop_na(value) %>%
  mutate(
    metric = str_match(metric_category, "^(.*) \\(")[, 2] %>%
      as.factor() %>%
      (function(x) fct_relevel(x, mixedsort(levels(x))))(),
    category = str_match(metric_category, "\\((.*)\\)$")[, 2] %>%
      as.factor() %>%
      fct_relevel(c("0/2 unseen", "1/2 unseen", "2/2 unseen", "1/1 unseen"))
  )
str(summary)

# %%
options(repr.plot.width = 6, repr.plot.height = 3.5)
gp <- ggplot(
  data = summary,
  mapping = aes(x = category, y = value, color = lik)
) +
  facet_grid(
    metric ~ dsgn_sb,
    scales = "free_y",
    labeller = labeller(
      metric = function(x) paste("Metric:", x),
      dsgn_sb = function(x) paste("Optimize scale/bias:", x)
    )
  ) +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.5),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.5),
  ) +
  labs(x = "Test category", y = "Metric value") +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_dsgn_train_double.pdf", gp,
  width = 6, height = 3.5
)
gp

# %% [markdown]
# ## Test

# %%
summary <- summary_all %>%
  filter(ds %in% double_ds, phs == "test") %>%
  melt(variable.name = "metric_category", value.name = "value") %>%
  drop_na(value) %>%
  mutate(
    metric = str_match(metric_category, "^(.*) \\(")[, 2] %>%
      as.factor() %>%
      (function(x) fct_relevel(x, mixedsort(levels(x))))(),
    category = str_match(metric_category, "\\((.*)\\)$")[, 2] %>%
      as.factor() %>%
      fct_relevel(c("0/2 unseen", "1/2 unseen", "2/2 unseen", "1/1 unseen"))
  )
str(summary)

# %%
options(repr.plot.width = 6, repr.plot.height = 3.5)
gp <- ggplot(
  data = summary,
  mapping = aes(x = category, y = value, color = lik)
) +
  facet_grid(
    metric ~ dsgn_sb,
    scales = "free_y",
    labeller = labeller(
      metric = function(x) paste("Metric:", x),
      dsgn_sb = function(x) paste("Optimize scale/bias:", x)
    )
  ) +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.5),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.5),
  ) +
  labs(x = "Test category", y = "Metric value") +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_dsgn_test_double.pdf", gp,
  width = 6, height = 3.5
)
gp
