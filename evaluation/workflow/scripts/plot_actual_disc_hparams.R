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
  library(yaml)
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
summary_all <- read_csv(
  "../../sum/actual_disc_hparams_resp.csv",
  col_type = c(
    ds = "f", scf = "f", aux = "f",
    sps = "f", acyc = "f", lik = "f", phs = "f"
  )
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
  df_list[[param]] <- df_filter %>% rename(value = !!as.symbol(param))
}
summary_all <- bind_rows(df_list, .id = "param") %>% mutate(
  param = param %>%
    fct_recode(!!!display$naming$params) %>%
    fct_relevel(names(display$naming$params)),
  value = value %>% fct_relevel(as.character(sort(as.numeric(levels(value)))))
)

# %% [markdown]
# # Training

# %%
options(repr.plot.width = 6, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "train"),
  mapping = aes(x = value, y = resp_dist, color = ds, group = ds)
) +
  facet_wrap(~param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  labs(x = "Hyperparameter value", y = "Cohen'd", col = "Dataset") +
  ggplot_theme(legend.position = "bottom")
ggplot_save(
  "../../sum/actual_disc_hparams_resp_train_dist.pdf", gp,
  width = 6, height = 4.5
)
gp

# %%
options(repr.plot.width = 6, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "train"),
  mapping = aes(x = value, y = resp_dist_diff, color = ds, group = ds)
) +
  facet_wrap(~param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  labs(
    x = "Hyperparameter value",
    y = "Directional difference in Cohen'd",
    col = "Dataset"
  ) +
  ggplot_theme(legend.position = "bottom")
ggplot_save(
  "../../sum/actual_disc_hparams_resp_train_dist_diff.pdf", gp,
  width = 6, height = 4.5
)
gp

# %%
options(repr.plot.width = 6, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "train"),
  mapping = aes(x = value, y = resp_acc, color = ds, group = ds)
) +
  facet_wrap(~param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  labs(
    x = "Hyperparameter value",
    y = "Responsiveness accuracy",
    col = "Dataset"
  ) +
  ggplot_theme(legend.position = "bottom")
ggplot_save(
  "../../sum/actual_disc_hparams_resp_train_acc.pdf", gp,
  width = 6, height = 4.5
)
gp

# %% [markdown]
# # Testing

# %%
options(repr.plot.width = 6, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "test"),
  mapping = aes(x = value, y = resp_dist, color = ds, group = ds)
) +
  facet_wrap(~param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  labs(x = "Hyperparameter value", y = "Cohen'd", col = "Dataset") +
  ggplot_theme(legend.position = "bottom")
ggplot_save(
  "../../sum/actual_disc_hparams_resp_test_dist.pdf", gp,
  width = 6, height = 4.5
)
gp

# %%
options(repr.plot.width = 6, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "test"),
  mapping = aes(x = value, y = resp_dist_diff, color = ds, group = ds)
) +
  facet_wrap(~param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  labs(
    x = "Hyperparameter value",
    y = "Directional difference in Cohen'd",
    col = "Dataset"
  ) +
  ggplot_theme(legend.position = "bottom")
ggplot_save(
  "../../sum/actual_disc_hparams_resp_test_dist_diff.pdf", gp,
  width = 6, height = 4.5
)
gp

# %%
options(repr.plot.width = 6, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "test"),
  mapping = aes(x = value, y = resp_acc, color = ds, group = ds)
) +
  facet_wrap(~param, scales = "free") +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 2) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  labs(
    x = "Hyperparameter value",
    y = "Responsiveness accuracy",
    col = "Dataset"
  ) +
  ggplot_theme(legend.position = "bottom")
ggplot_save(
  "../../sum/actual_disc_hparams_resp_test_acc.pdf", gp,
  width = 6, height = 4.5
)
gp
