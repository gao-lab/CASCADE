.r_profile_user <- Sys.getenv("R_PROFILE_USER")
if (.r_profile_user != "") {
  .cwd <- getwd()
  setwd(dirname(.r_profile_user))
  source("renv/activate.R")
  setwd(.cwd)
}


ggplot_theme <- function(...) {
  return(ggplot2::theme(
    plot.background = ggplot2::element_blank(),
    panel.grid.major = ggplot2::element_line(
      color = "#EEEEEE", linetype = "longdash"
    ),
    panel.grid.minor = ggplot2::element_blank(),
    panel.background = ggplot2::element_rect(fill = "#FFFFFF"),
    legend.background = ggplot2::element_blank(),
    legend.box.background = ggplot2::element_blank(),
    axis.line = ggplot2::element_line(color = "#000000"),
    ...
  ))
}


ggplot_save <- function(filename, ...) {
  ggplot2::ggsave(filename, ..., dpi = 600, bg = "transparent")
}
