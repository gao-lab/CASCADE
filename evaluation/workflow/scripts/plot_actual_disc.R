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
display <- read_yaml("../../config/display.yaml")
names(display$naming$datasets) <- gsub(
  "<br>", "\n", names(display$naming$datasets)
)

# %%
summary_all <- read_csv(
  "../../sum/actual_disc_resp.csv",
  col_type = c(ds = "f", scf = "f", aux = "f", meth = "f", run = "f", phs = "f")
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
      fct_recode(!!!display$naming$methods) %>%
      fct_relevel(names(display$naming$methods))
  ) %>%
  sample_frac()
str(summary_all)

# %% [markdown]
# # Train

# %%
options(repr.plot.width = 5, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "train", imp == "20"),
  mapping = aes(x = meth, y = resp_dist, col = meth)
) +
  facet_wrap(~ds, scales = "free_y") +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.4),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.4),
  ) +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(
    x = "Dataset", y = "Cohen's d", col = "Method"
  ) +
  guides(color = "none") +
  ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggplot_save(
  "../../sum/actual_disc_resp_train_dist.pdf", gp,
  width = 5, height = 4.5
)
gp

# %%
options(repr.plot.width = 5, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "train", imp == "20"),
  mapping = aes(x = meth, y = resp_dist_diff, col = meth)
) +
  facet_wrap(~ds, scales = "free_y") +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.4),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.4),
  ) +
  geom_hline(yintercept = 0.0, linetype = "dashed", color = "darkred") +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(
    x = "Dataset", y = "Directional difference in Cohen's d", col = "Method"
  ) +
  guides(color = "none") +
  ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggplot_save(
  "../../sum/actual_disc_resp_train_dist_diff.pdf", gp,
  width = 5, height = 4.5
)
gp

# %%
options(repr.plot.width = 5, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "train", imp == "20"),
  mapping = aes(x = meth, y = resp_acc, col = meth)
) +
  facet_wrap(~ds, scales = "free_y") +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.4),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.4),
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "darkred") +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(
    x = "Dataset", y = "Responsiveness accuracy", col = "Method"
  ) +
  guides(color = "none") +
  ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggplot_save(
  "../../sum/actual_disc_resp_train_acc.pdf", gp,
  width = 5, height = 4.5
)
gp

# %% [markdown]
# # Test

# %%
options(repr.plot.width = 5, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "test", imp == "20"),
  mapping = aes(x = meth, y = resp_dist, col = meth)
) +
  facet_wrap(~ds, scales = "free_y") +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.4),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.4),
  ) +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(
    x = "Dataset", y = "Cohen's d", col = "Method"
  ) +
  guides(color = "none") +
  ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggplot_save(
  "../../sum/actual_disc_resp_test_dist.pdf", gp,
  width = 5, height = 4.5
)
gp

# %%
options(repr.plot.width = 5, repr.plot.height = 4.5)
gp <- ggplot(
  data = summary_all %>% filter(phs == "test", imp == "20"),
  mapping = aes(x = meth, y = resp_dist_diff, col = meth)
) +
  facet_wrap(~ds, scales = "free_y") +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.4),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.4),
  ) +
  geom_hline(yintercept = 0.0, linetype = "dashed", color = "darkred") +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(
    x = "Dataset", y = "Directional difference in Cohen's d", col = "Method"
  ) +
  guides(color = "none") +
  ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggplot_save(
  "../../sum/actual_disc_resp_test_dist_diff.pdf", gp,
  width = 5, height = 4.5
)
gp

# %%
options(repr.plot.width = 4.5, repr.plot.height = 4)
gp <- ggplot(
  data = summary_all %>% filter(phs == "test", imp == "20"),
  mapping = aes(x = meth, y = resp_acc, col = meth)
) +
  facet_wrap(~ds, scales = "free_y") +
  stat_summary(
    fun = mean, geom = "point",
    size = 2.5, position = position_dodge(width = 0.4),
  ) +
  stat_summary(
    fun.data = mean_se, geom = "errorbar",
    linewidth = 1, width = 0, position = position_dodge(width = 0.4),
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "darkred") +
  scale_color_manual(values = unlist(display$palette$methods)) +
  labs(
    x = "Dataset", y = "Responsiveness accuracy", col = "Method"
  ) +
  guides(color = "none") +
  ggplot_theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggplot_save(
  "../../sum/actual_disc_resp_test_acc.pdf", gp,
  width = 4.5, height = 4
)
gp

# %%
