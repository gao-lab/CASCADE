# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
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
  library(readr)
  library(reshape2)
  library(yaml)
})

# %%
display <- read_yaml("../../config/display.yaml")

# %%
summary_all <- read_csv(
  "../../sum/sim_disc_sub.csv",
  col_types = c(gph_type = "f", act = "f", meth = "f", run = "f")
) %>%
  mutate(
    n_vars = n_vars %>% as.factor(),
    in_deg = in_deg %>% as.factor(),
    gph_sd = gph_sd %>% as.factor(),
    n_obs = n_obs %>% as.factor(),
    int_frac = int_frac %>%
      as.factor() %>%
      fct_relabel(function(x) sprintf("%d%%", as.numeric(x) * 100)),
    snr = snr %>% as.factor(),
    dat_sd = dat_sd %>% as.factor(),
    unobs_frac = (1 - sub_frac) %>%
      as.factor() %>%
      fct_relabel(function(x) sprintf("%d%%", as.numeric(x) * 100)),
    sub_frac = sub_frac %>%
      as.factor() %>%
      fct_relabel(function(x) sprintf("%d%%", as.numeric(x) * 100)),
    sub_sd = sub_sd %>% as.factor(),
    aux = aux %>% factor(
      levels = c("nil", "k=10", "k=20", "k=40"),
      labels = c("False", "k=10", "k=20", "True")
    ),
    kg = kg %>% as.factor(),
    kc = kc %>% as.factor(),
    div_sd = div_sd %>% as.factor(),
    meth = meth %>%
      fct_recode(!!!display$naming$methods) %>%
      fct_relevel(names(display$naming$methods)),
    log10_shd = log10(shd + 1),
    log10_sid = log10(sid + 1)
  ) %>%
  sample_frac()
str(summary_all)

# %%
summary <- summary_all %>%
  filter(
    (meth != "CASCADE") |
      (
        run == paste0(
          "sparse=L1-acyc=SpecNorm-n_particles=1-dim_u=2-dim_v=64-dropout=0.2-",
          "lam=0.1-weight_decay=0.01-tune_ctfact=True-run_sd=0"
        )
      ),
    aux %in% c("False", "True")
  ) %>%
  melt(variable.name = "metric", value.name = "value") %>%
  mutate(
    metric = metric %>%
      fct_recode(!!!display$naming$metrics) %>%
      fct_relevel(names(display$naming$metrics))
  )
str(summary)

# %% [markdown]
# # Primary metrics

# %%
primary_metrics <- c("log10 SHD", "Avg precision")

# %%
options(repr.plot.width = 8, repr.plot.height = 3.6)
gp <- ggplot(
  data = summary %>% filter(metric %in% primary_metrics),
  mapping = aes(
    x = unobs_frac, y = value, col = meth, linetype = aux,
    group = interaction(meth, aux)
  )
) +
  facet_wrap(
    ~metric,
    ncol = 3, scales = "free_y",
    labeller = labeller(metric = function(x) paste("Metric:", x))
  ) +
  geom_point() +
  geom_line() +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(
    x = "Unobserved fraction", y = "Metric value",
    col = "Method", linetype = "Latent vars"
  ) +
  ggplot_theme()
ggplot_save("../../sum/sim_disc_sub_primary.pdf", gp, width = 8, height = 3.6)
gp

# %% [markdown]
# # Secondary metrics

# %%
secondary_metrics <- c(
  "log10 SID", "Accuracy", "F1 score", "Precision", "Recall", "AUROC"
)

# %%
options(repr.plot.width = 8, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary %>% filter(metric %in% secondary_metrics),
  mapping = aes(
    x = unobs_frac, y = value, col = meth, linetype = aux,
    group = interaction(meth, aux)
  )
) +
  facet_wrap(
    ~metric,
    ncol = 3, scales = "free_y",
    labeller = labeller(metric = function(x) paste("Metric:", x))
  ) +
  geom_point() +
  geom_line() +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(
    x = "Unobserved fraction", y = "Metric value",
    col = "Method", linetype = "Latent vars"
  ) +
  ggplot_theme()
ggplot_save("../../sum/sim_disc_sub_secondary.pdf", gp, width = 8, height = 4.5)
gp
