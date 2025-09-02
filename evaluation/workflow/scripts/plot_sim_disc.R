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
  library(patchwork)
  library(readr)
  library(reshape2)
  library(yaml)
})

# %%
display <- read_yaml("../../config/display.yaml")

# %%
summary_all <- read_csv(
  "../../sum/sim_disc_true.csv",
  col_types = c(gph_tp = "f", act = "f", meth = "f", run = "f")
) %>%
  mutate(
    n_vars = n_vars %>% as.factor(),
    in_deg = in_deg %>% as.factor(),
    gph_sd = gph_sd %>% as.factor(),
    n_obs = n_obs %>% as.factor(),
    int_f = int_f %>%
      as.factor() %>%
      fct_relabel(function(x) sprintf("%d%%", as.numeric(x) * 100)),
    snr = snr %>% as.factor(),
    dat_sd = dat_sd %>% as.factor(),
    sub_f = sub_f %>% as.factor(),
    sub_sd = sub_sd %>% as.factor(),
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
  melt(variable.name = "metric", value.name = "value") %>%
  mutate(
    metric = metric %>%
      fct_recode(!!!display$naming$metrics) %>%
      fct_relevel(names(display$naming$metrics))
  )
str(summary)

# %%
primary_metrics <- c("log10 SHD", "Avg precision")
secondary_metrics <- c("AUROC", "Precision", "Recall")
metrics <- c(primary_metrics, secondary_metrics)

# %% [markdown]
# # Overview

# %%
options(repr.plot.width = 6, repr.plot.height = 4.5)
gps <- list()
for (primary_metric in primary_metrics) {
  gps[[length(gps) + 1]] <- ggplot(
    data = summary %>% filter(metric == primary_metric),
    mapping = aes(x = n_vars, y = value, col = meth, group = meth)
  ) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun = mean, geom = "point", size = 1.5) +
    stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
    scale_color_manual(values = unlist(display$palette$methods)) +
    labs(x = "No. of variables", y = primary_metric, col = "Method") +
    ggplot_theme()
}
for (primary_metric in primary_metrics) {
  gps[[length(gps) + 1]] <- ggplot(
    data = summary %>% filter(
      metric == primary_metric, as.numeric(as.character(n_vars)) < 80
    ), mapping = aes(x = int_f, y = value, col = meth, group = meth)
  ) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun = mean, geom = "point", size = 1.5) +
    stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
    scale_color_manual(values = unlist(display$palette$methods)) +
    labs(x = "Intervention fraction", y = primary_metric, col = "Method") +
    ggplot_theme()
}
gp <- wrap_plots(gps, nrow = 2, byrow = FALSE) +
  plot_layout(guides = "collect") & theme(legend.position = "right")
ggplot_save(
  "../../sum/sim_disc_true_overview.pdf", gp,
  width = 6, height = 4.5
)
gp

# %% [markdown]
# # Detailed

# %% [markdown]
# ## Number of variables

# %%
options(repr.plot.width = 10, repr.plot.height = 10)
gps <- list()
for (any_metric in metrics) {
  if (length(gps) == 0) {
    row_theme <- ggplot_theme(
      axis.title.x = element_blank(),
      strip.text = element_text(size = 11, margin = margin(b = 7)),
      strip.background = element_blank()
    )
  } else if (length(gps) == length(metrics) - 1) {
    row_theme <- ggplot_theme(
      axis.title.x = element_text(size = 12, margin = margin(t = 7)),
      strip.text = element_blank(),
      strip.background = element_blank()
    )
  } else {
    row_theme <- ggplot_theme(
      axis.title.x = element_blank(),
      strip.text = element_blank(),
      strip.background = element_blank()
    )
  }
  gps[[length(gps) + 1]] <- ggplot(
    data = summary %>% filter(metric == any_metric),
    mapping = aes(x = n_vars, y = value, col = meth, group = meth)
  ) +
    facet_wrap(
      ~int_f,
      nrow = 1, scales = "free_y",
      labeller = as_labeller(function(x) paste("Intervened:", x))
    ) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun = mean, geom = "point", size = 1.5) +
    stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
    scale_color_manual(values = unlist(display$palette$methods)) +
    labs(x = "No. of variables", y = any_metric, col = "Method") +
    row_theme
}
gp <- wrap_plots(gps, ncol = 1) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggplot_save(
  "../../sum/sim_disc_true_n_vars.pdf", gp,
  width = 10, height = 10
)
gp

# %% [markdown]
# ## Intervention fraction

# %%
options(repr.plot.width = 10, repr.plot.height = 10)
gps <- list()
metrics <- c(primary_metrics, secondary_metrics)
for (any_metric in metrics) {
  if (length(gps) == 0) {
    row_theme <- ggplot_theme(
      axis.title.x = element_blank(),
      strip.text = element_text(size = 11, margin = margin(b = 7)),
      strip.background = element_blank()
    )
  } else if (length(gps) == length(metrics) - 1) {
    row_theme <- ggplot_theme(
      axis.title.x = element_text(size = 12, margin = margin(t = 7)),
      strip.text = element_blank(),
      strip.background = element_blank()
    )
  } else {
    row_theme <- ggplot_theme(
      axis.title.x = element_blank(),
      strip.text = element_blank(),
      strip.background = element_blank()
    )
  }
  gps[[length(gps) + 1]] <- ggplot(
    data = summary %>% filter(metric == any_metric),
    mapping = aes(x = int_f, y = value, col = meth, group = meth)
  ) +
    facet_wrap(
      ~n_vars,
      nrow = 1, scales = "free_y",
      labeller = as_labeller(function(x) paste("No. of variables:", x))
    ) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun = mean, geom = "point", size = 1.5) +
    stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
    scale_color_manual(values = unlist(display$palette$methods)) +
    labs(x = "Intervened fraction", y = any_metric, col = "Method") +
    row_theme
}
gp <- wrap_plots(gps, ncol = 1) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggplot_save(
  "../../sum/sim_disc_true_int_f.pdf", gp,
  width = 10, height = 10
)
gp
