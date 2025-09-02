include: "utils.smk"


assert config["scf"][0] == "cpl"


def per_meth(meth, files, complete_scf_only=True):
    cfg = {
        k: v[0] if complete_scf_only and k == "scf" else v for k, v in config.items()
    }
    return target_files(
        expand(
            join(
                "inf",
                conf_pattern(cfg["ctx"], name="ctx"),
                conf_pattern(cfg["dat"], name="dat"),
                conf_pattern(cfg["sub"], name="sub"),
                conf_pattern(cfg["scf"], name="scf"),
                conf_pattern(cfg["aux"], name="aux"),
                conf_pattern(cfg["div"], name="div"),
                meth,
                conf_pattern(cfg["meth"][meth], name="run"),
            ),
            **conf_values(cfg["ctx"], name="ctx"),
            **conf_values(cfg["dat"], name="dat"),
            **conf_values(cfg["sub"], name="sub"),
            **conf_values(cfg["scf"], name="scf"),
            **conf_values(cfg["aux"], name="aux"),
            **conf_values(cfg["div"], name="div"),
            **conf_values(cfg["meth"][meth], name="run"),
        ),
        files,
    )


rule plot:
    input:
        "sum/sim_scale_time.csv",
        "sum/sim_scale_metrics.csv",
    output:
        "sum/sim_scale_time.pdf",
        "sum/sim_scale_metrics.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_sim_scale.R"


rule summarize_time:
    input:
        reduce(
            add,
            (
                per_meth(
                    meth,
                    (
                        ["info_disc.yaml", "info_acyc.yaml"]
                        if meth == "cascade"
                        else ["info.yaml"]
                    ),
                    complete_scf_only=meth not in {"pc", "ges", "gies", "cascade"},
                )
                for meth in config["meth"]
            ),
        ),
    output:
        "sum/sim_scale_time.csv",
    params:
        pattern=lambda wildcards: join(
            "inf",
            conf_pattern(config["ctx"], name="ctx"),
            conf_pattern(config["dat"], name="dat"),
            conf_pattern(config["sub"], name="sub"),
            conf_pattern(config["scf"], name="scf"),
            conf_pattern(config["aux"], name="aux"),
            conf_pattern(config["div"], name="div"),
            "{meth}",
            "{run}",
            "{info}.yaml",
        ),
    localrule: True
    script:
        "../scripts/summarize.py"


rule summarize_metrics:
    input:
        reduce(
            add,
            (
                per_meth(
                    meth,
                    ["metrics_disc_true.yaml"],
                    complete_scf_only=meth not in {"pc", "ges", "gies", "cascade"},
                )
                for meth in config["meth"]
            ),
        ),
    output:
        "sum/sim_scale_metrics.csv",
    params:
        pattern=lambda wildcards: join(
            "inf",
            conf_pattern(config["ctx"], name="ctx"),
            conf_pattern(config["dat"], name="dat"),
            conf_pattern(config["sub"], name="sub"),
            conf_pattern(config["scf"], name="scf"),
            conf_pattern(config["aux"], name="aux"),
            conf_pattern(config["div"], name="div"),
            "{meth}",
            "{run}",
            "metrics_disc_true.yaml",
        ),
    localrule: True
    script:
        "../scripts/summarize.py"
