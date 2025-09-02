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

# %% editable=true slideshow={"slide_type": ""}
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
pattern <- "^([a-z_]+)(?:_top(\\d+))? \\((.+)\\)$"

# %%
summary_all <- read_csv(
  "../../sum/actual_ctfact_ablt.csv",
  col_type = c(ds = "f", scf = "f", aux = "f", ablt = "f", phs = "f")
) %>%
  select(
    -starts_with("true_"), -starts_with("pred_"),
    -starts_with("dir_"), -starts_with("edist")
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
    ablt = ablt %>%
      fct_recode(!!!display$naming$ablation) %>%
      fct_relevel(names(display$naming$ablation)),
  ) %>%
  sample_frac()
str(summary_all)

# %% [markdown]
# # Single

# %%
summary <- summary_all %>%
  melt(variable.name = "metric_category", value.name = "value") %>%
  drop_na(value) %>%
  mutate(
    metric = str_match(metric_category, pattern)[, 2] %>%
      as.factor() %>%
      fct_recode(!!!display$naming$metrics) %>%
      fct_relevel(names(display$naming$metrics)),
    top_de = str_match(metric_category, pattern)[, 3] %>%
      as.integer() %>%
      as.factor() %>%
      fct_explicit_na(na_level = "all"),
    category = str_match(metric_category, pattern)[, 4] %>% as.factor()
  ) %>%
  filter(ablt != "No intervention", category %in% c("1/1 unseen", "1 seen"))
str(summary)

# %%
options(repr.plot.width = 6.2, repr.plot.height = 4)
gp <- ggplot(
  data = summary %>% filter(phs == "train"),
  mapping = aes(x = top_de, y = value, col = ablt, group = ablt)
) +
  facet_grid(metric ~ ds, scales = "free_y") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 1.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_manual(values = unlist(display$palette$ablation)) +
  labs(x = "No. of top DEGs", y = "Metric value", col = "Ablation") +
  ggplot_theme(
    axis.text.x = element_text(angle = 40, hjust = 1),
    legend.position = "bottom"
  )
ggplot_save(
  "../../sum/actual_ctfact_ablt_train_single.pdf", gp,
  width = 6, height = 4
)
gp

# %%
options(repr.plot.width = 6.2, repr.plot.height = 4)
gp <- ggplot(
  data = summary %>% filter(phs == "test"),
  mapping = aes(x = top_de, y = value, col = ablt, group = ablt)
) +
  facet_grid(metric ~ ds, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 1.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_manual(values = unlist(display$palette$ablation)) +
  labs(x = "No. of top DEGs", y = "Metric value", col = "Ablation") +
  ggplot_theme(
    axis.text.x = element_text(angle = 40, hjust = 1),
    legend.position = "bottom"
  )
ggplot_save(
  "../../sum/actual_ctfact_ablt_test_single.pdf", gp,
  width = 6, height = 4
)
gp

# %% [markdown]
# # Double

# %%
double_ds <- c("Norman-2019")
summary <- summary_all %>%
  filter(ds %in% double_ds) %>%
  melt(variable.name = "metric_category", value.name = "value") %>%
  drop_na(value) %>%
  mutate(
    metric = str_match(metric_category, pattern)[, 2] %>%
      as.factor() %>%
      fct_recode(!!!display$naming$metrics) %>%
      fct_relevel(names(display$naming$metrics)),
    top_de = str_match(metric_category, pattern)[, 3] %>%
      as.integer() %>%
      as.factor() %>%
      fct_explicit_na(na_level = "all"),
    category = str_match(metric_category, pattern)[, 4] %>%
      as.factor() %>%
      fct_relevel(c("0/2 unseen", "1/2 unseen", "2/2 unseen", "1/1 unseen"))
  ) %>%
  filter(
    ablt != "No intervention",
    category %in% c("0/2 unseen", "1/2 unseen", "2/2 unseen", "2 seen")
  )
str(summary)

# %%
options(repr.plot.width = 1.9, repr.plot.height = 3.5)
gp <- ggplot(
  data = summary %>% filter(phs == "train"),
  mapping = aes(x = top_de, y = value, color = ablt, group = ablt)
) +
  facet_grid(metric ~ category, scales = "free_y") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 1.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_manual(values = unlist(display$palette$ablation)) +
  labs(x = "No. of top DEGs", y = "Metric value", col = "Ablation") +
  guides(color = "none") +
  ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_ablt_train_double.pdf", gp,
  width = 1.9, height = 3.5
)
gp

# %%
options(repr.plot.width = 4, repr.plot.height = 3.5)
gp <- ggplot(
  data = summary %>% filter(phs == "test"),
  mapping = aes(x = top_de, y = value, color = ablt, group = ablt)
) +
  facet_grid(metric ~ category, scales = "free_y") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 1.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_manual(values = unlist(display$palette$ablation)) +
  labs(x = "No. of top DEGs", y = "Metric value", col = "Ablation") +
  guides(color = "none") +
  ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_ablt_test_double.pdf", gp,
  width = 4, height = 3.5
)
gp
