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
  library(viridis)
  library(yaml)
})

# %%
display <- read_yaml("../../config/display.yaml")

# %%
summary_all <- read_csv(
  "../../sum/sim_disc_scf_true.csv",
  col_types = c(gph_type = "f", act = "f", meth = "f", run = "f")
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
    sub_f = sub_f %>%
      as.factor() %>%
      fct_relabel(function(x) sprintf("%d%%", as.numeric(x) * 100)),
    sub_sd = sub_sd %>% as.factor(),
    tpr = tpr %>% as.factor() %>% fct_rev(),
    fpr = fpr %>% as.factor(),
    scf_sd = scf_sd %>% as.factor(),
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
head(summary)

# %% [markdown]
# # Heatmaps

# %%
options(repr.plot.width = 4.7, repr.plot.height = 4)
gp <- ggplot(
  data = summary %>% filter(metric == "log10 SHD"),
  mapping = aes(x = fpr, y = tpr, z = value)
) +
  facet_wrap(~meth, scales = "free") +
  stat_summary_2d(fun = mean) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  scale_fill_viridis(direction = -1) +
  labs(x = "Scaffold FPR", y = "Scaffold TPR", fill = "log10 SHD") +
  ggplot_theme() +
  theme(panel.grid.major = element_blank(), axis.line = element_blank())
ggplot_save(
  "../../sum/sim_disc_scf_true_heatmap_shd.pdf", gp,
  width = 4.7, height = 4
)
gp

# %%
options(repr.plot.width = 4.7, repr.plot.height = 4)
gp <- ggplot(
  data = summary %>% filter(metric == "Avg precision"),
  mapping = aes(x = fpr, y = tpr, z = value)
) +
  facet_wrap(~meth, scales = "free") +
  stat_summary_2d(fun = mean) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  scale_fill_viridis() +
  labs(x = "Scaffold FPR", y = "Scaffold TPR", fill = "Avg precision") +
  ggplot_theme() +
  theme(panel.grid.major = element_blank(), axis.line = element_blank())
ggplot_save(
  "../../sum/sim_disc_scf_true_heatmap_ap.pdf", gp,
  width = 4.7, height = 4
)
gp

# %% [markdown]
# # Line plots

# %%
metrics <- c("log10 SHD", "Avg precision", "AUROC", "Precision", "Recall")

# %%
options(repr.plot.width = 7, repr.plot.height = 8.5)
gps <- list()
for (any_metric in metrics) {
  if (length(gps) == 0) {
    row_theme <- ggplot_theme(
      axis.title.x = element_blank(),
      strip.text = element_text(size = 11, margin = margin(b = 8)),
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
    mapping = aes(x = fpr, y = value, col = meth, group = meth)
  ) +
    facet_wrap(
      ~tpr,
      nrow = 1, scales = "free_y",
      labeller = as_labeller(function(x) paste("Scaffold TPR:", x))
    ) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun = mean, geom = "point", size = 1.5) +
    stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
    scale_color_manual(values = unlist(display$palette$methods)) +
    labs(x = "Scaffold FPR", y = any_metric, col = "Method") +
    row_theme
}
gp <- wrap_plots(gps, ncol = 1) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggplot_save(
  "../../sum/sim_disc_scf_true_lines.pdf", gp,
  width = 7, height = 8.5
)
gp
