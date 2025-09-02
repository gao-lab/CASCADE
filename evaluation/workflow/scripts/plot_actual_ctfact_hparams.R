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
  library(viridis)
})

# %%
params <- read_yaml("../../config/config.yaml")$actual_disc_hparams$meth$cascade
defaults <- lapply(params, function(x) ifelse(is.list(x), x$default, x))

# %%
display <- read_yaml("../../config/display.yaml")
names(display$naming$datasets) <- gsub(
  "<br>", "\n", names(display$naming$datasets)
)

# %%
pattern <- "^([a-z_]+)(?:_top(\\d+))? \\((.+)\\)$"

# %%
summary_all <- read_csv(
  "../../sum/actual_ctfact_hparams.csv",
  col_type = c(
    ds = "f", scf = "f", aux = "f",
    sps = "f", acyc = "f", lik = "f", phs = "f"
  )
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
    scf = scf %>%
      fct_recode(!!!display$naming$scf) %>%
      fct_relevel(names(display$naming$scf)),
    kg = factor(kg),
    kc = factor(kc),
    div_sd = factor(div_sd),
    nptc = factor(nptc),
    dz = factor(dz),
    beta = factor(beta),
    lam = factor(lam),
    alp = factor(alp),
    tune_ct = factor(tune_ct),
    run_sd = factor(run_sd),
  ) %>%
  sample_frac()
str(summary_all)

# %% [markdown]
# # Organizing

# %%
df_list <- list()
for (param in names(params)) {
  if (!is.list(params[[param]])) next
  df_filter <- summary_all
  for (other in names(params)) {
    if (other != param) {
      df_filter <- df_filter %>%
        filter(!!as.symbol(other) == defaults[[other]])
    }
  }
  df_list[[param]] <- df_filter %>% rename(param_val = !!as.symbol(param))
}
summary_all <- bind_rows(df_list, .id = "param") %>% mutate(
  param = param %>%
    fct_recode(!!!display$naming$params) %>%
    fct_relevel(names(display$naming$params)),
  param_val = param_val %>% fct_relevel(as.character(sort(as.numeric(levels(param_val)))))
)

# %% [markdown]
# # Single

# %%
summary <- summary_all %>%
  melt(variable.name = "metric_category", value.name = "metric_val") %>%
  drop_na(metric_val) %>%
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
  filter(category %in% c("1/1 unseen", "1 seen"))
str(summary)

# %%
options(repr.plot.width = 10, repr.plot.height = 7)
gp <- ggplot(
  data = summary %>% filter(metric == "Delta correlation", phs == "train"),
  mapping = aes(
    x = param_val, y = metric_val, color = top_de, group = top_de
  )
) +
  facet_grid(ds ~ param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_viridis(discrete = TRUE, direction = -1) +
  labs(
    x = "Hyperparameter value",
    y = "Delta correlation",
    col = "Number of\nDE genes"
  ) +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_hparams_train_delta_pcc_single.pdf", gp,
  width = 10, height = 7
)
gp

# %%
options(repr.plot.width = 10, repr.plot.height = 7)
gp <- ggplot(
  data = summary %>% filter(metric == "Normalized MSE", phs == "train"),
  mapping = aes(
    x = param_val, y = metric_val, color = top_de, group = top_de
  )
) +
  facet_grid(ds ~ param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_viridis(discrete = TRUE, direction = -1) +
  labs(
    x = "Hyperparameter value",
    y = "Normalized MSE",
    col = "Number of\nDE genes"
  ) +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_hparams_train_normalized_mse_single.pdf", gp,
  width = 10, height = 7
)
gp

# %%
options(repr.plot.width = 10, repr.plot.height = 7)
gp <- ggplot(
  data = summary %>% filter(metric == "Delta correlation", phs == "test"),
  mapping = aes(
    x = param_val, y = metric_val, color = top_de, group = top_de
  )
) +
  facet_grid(ds ~ param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_viridis(discrete = TRUE, direction = -1) +
  labs(
    x = "Hyperparameter value",
    y = "Delta correlation",
    col = "Number of\nDE genes"
  ) +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_hparams_test_delta_pcc_single.pdf", gp,
  width = 10, height = 7
)
gp

# %%
options(repr.plot.width = 10, repr.plot.height = 7)
gp <- ggplot(
  data = summary %>% filter(metric == "Normalized MSE", phs == "test"),
  mapping = aes(
    x = param_val, y = metric_val, color = top_de, group = top_de
  )
) +
  facet_grid(ds ~ param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_viridis(discrete = TRUE, direction = -1) +
  labs(
    x = "Hyperparameter value",
    y = "Normalized MSE",
    col = "Number of\nDE genes"
  ) +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_hparams_test_normalized_mse_single.pdf", gp,
  width = 10, height = 7
)
gp

# %% [markdown]
# # Double

# %%
double_ds <- c("Norman-2019")
summary <- summary_all %>%
  filter(ds %in% double_ds) %>%
  melt(variable.name = "metric_category", value.name = "metric_val") %>%
  drop_na(metric_val) %>%
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
    category %in% c("0/2 unseen", "1/2 unseen", "2/2 unseen", "2 seen")
  )

# %%
options(repr.plot.width = 10, repr.plot.height = 2.4)
gp <- ggplot(
  data = summary %>% filter(metric == "Delta correlation", phs == "train"),
  mapping = aes(
    x = param_val, y = metric_val, color = top_de, group = top_de
  )
) +
  facet_grid(category ~ param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_viridis(discrete = TRUE, direction = -1) +
  labs(
    x = "Hyperparameter value",
    y = "Delta correlation",
    col = "Number of\nDE genes"
  ) +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_hparams_train_delta_pcc_double.pdf", gp,
  width = 10, height = 2.4
)
gp

# %%
options(repr.plot.width = 10, repr.plot.height = 2.4)
gp <- ggplot(
  data = summary %>% filter(metric == "Normalized MSE", phs == "train"),
  mapping = aes(
    x = param_val, y = metric_val, color = top_de, group = top_de
  )
) +
  facet_grid(category ~ param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_viridis(discrete = TRUE, direction = -1) +
  labs(
    x = "Hyperparameter value",
    y = " Normalized MSE",
    col = "Number of\nDE genes"
  ) +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_hparams_train_normalized_mse_double.pdf", gp,
  width = 10, height = 2.4
)
gp

# %%
options(repr.plot.width = 10, repr.plot.height = 5)
gp <- ggplot(
  data = summary %>% filter(metric == "Delta correlation", phs == "test"),
  mapping = aes(
    x = param_val, y = metric_val, color = top_de, group = top_de
  )
) +
  facet_grid(category ~ param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_viridis(discrete = TRUE, direction = -1) +
  labs(
    x = "Hyperparameter value",
    y = "Delta correlation",
    col = "Number of\nDE genes"
  ) +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_hparams_test_delta_pcc_double.pdf", gp,
  width = 10, height = 5
)
gp

# %%
options(repr.plot.width = 10, repr.plot.height = 5)
gp <- ggplot(
  data = summary %>% filter(metric == "Normalized MSE", phs == "test"),
  mapping = aes(
    x = param_val, y = metric_val, color = top_de, group = top_de
  )
) +
  facet_grid(category ~ param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_color_viridis(discrete = TRUE, direction = -1) +
  labs(
    x = "Hyperparameter value",
    y = "Normalized MSE",
    col = "Number of\nDE genes"
  ) +
  ggplot_theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggplot_save(
  "../../sum/actual_ctfact_hparams_test_normalized_mse_double.pdf", gp,
  width = 10, height = 5
)
gp
