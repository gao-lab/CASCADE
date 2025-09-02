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

# %% [markdown]
# # Time

# %%
summary_all <- read_csv(
  "../../sum/sim_scale_time.csv",
  col_types = c(
    gph_tp = "f", act = "f", scf = "f", meth = "f", run = "f", info = "f"
  )
) %>%
  mutate(
    n_vars = n_vars %>% as.factor(),
    in_deg = in_deg %>% as.factor(),
    gph_sd = gph_sd %>% as.factor(),
    n_obs = n_obs %>% as.factor(),
    int_f = int_f %>% as.factor(),
    snr = snr %>% as.factor(),
    dat_sd = dat_sd %>% as.factor(),
    sub_f = sub_f %>% as.factor(),
    sub_sd = sub_sd %>% as.factor(),
    scf = scf %>%
      fct_relabel(
        function(x) ifelse(x == "cpl", "Complete", "Informative")
      ),
    kg = kg %>% as.factor(),
    kc = kc %>% as.factor(),
    div_sd = div_sd %>% as.factor(),
    meth = meth %>%
      fct_recode(!!!display$naming$methods) %>%
      fct_relevel(names(display$naming$methods)),
  ) %>%
  group_by(across(-c(info, time))) %>%
  summarize(time = sum(time)) %>%
  ungroup() %>%
  sample_frac()
str(summary_all)

# %%
options(repr.plot.width = 3.5, repr.plot.height = 3.5)
gp <- ggplot(
  data = summary_all,
  mapping = aes(
    x = n_vars, y = time, col = meth,
    linetype = scf, group = interaction(meth, scf)
  )
) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point", size = 1.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
  scale_y_log10() +
  scale_color_manual(values = unlist(display$palette$methods)) +
  scale_linetype_manual(values = c("solid", "dotted")) +
  labs(x = "No. of variables", y = "Time in secs", linetype = "Scaffold") +
  guides(color = "none") +
  ggplot_theme(legend.position = "bottom")
ggplot_save("../../sum/sim_scale_time.pdf", gp, width = 3.5, height = 3.5)
gp

# %% [markdown]
# # Metrics

# %%
summary_all <- read_csv(
  "../../sum/sim_scale_metrics.csv",
  col_types = c(
    gph_tp = "f", act = "f", scf = "f", meth = "f", run = "f"
  )
) %>%
  mutate(
    n_vars = n_vars %>% as.factor(),
    in_deg = in_deg %>% as.factor(),
    gph_sd = gph_sd %>% as.factor(),
    n_obs = n_obs %>% as.factor(),
    int_f = int_f %>% as.factor(),
    snr = snr %>% as.factor(),
    dat_sd = dat_sd %>% as.factor(),
    sub_f = sub_f %>% as.factor(),
    sub_sd = sub_sd %>% as.factor(),
    scf = scf %>%
      fct_relabel(
        function(x) ifelse(x == "cpl", "Complete", "Informative")
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
  melt(variable.name = "metric", value.name = "value") %>%
  mutate(
    metric = metric %>%
      fct_recode(!!!display$naming$metrics) %>%
      fct_relevel(names(display$naming$metrics))
  )

# %%
primary_metrics <- c("log10 SHD", "Avg precision")
secondary_metrics <- c("AUROC", "Precision", "Recall")
metrics <- c(primary_metrics, secondary_metrics)

# %%
options(repr.plot.width = 10, repr.plot.height = 2.5)
gps <- list()
for (any_metric in metrics) {
  gps[[length(gps) + 1]] <- ggplot(
    data = summary %>% filter(metric == any_metric),
    mapping = aes(
      x = n_vars, y = value, col = meth,
      linetype = scf, group = interaction(meth, scf)
    )
  ) +
    stat_summary(fun = mean, geom = "line") +
    stat_summary(fun = mean, geom = "point", size = 1) +
    stat_summary(fun.data = mean_se, geom = "errorbar", linewidth = 1, width = 0) +
    scale_color_manual(values = unlist(display$palette$methods)) +
    scale_linetype_manual(values = c("solid", "dotted")) +
    labs(x = "No. of variables", y = any_metric, linetype = "Scaffold") +
    guides(color = "none") +
    ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
}
gp <- wrap_plots(gps, nrow = 1) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
ggplot_save("../../sum/sim_scale_metrics.pdf", gp, width = 10, height = 2.5)
gp
