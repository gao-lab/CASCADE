include: "utils.smk"


directories = reduce(
    add,
    map(
        lambda meth: expand(
            join(
                "inf",
                conf_pattern(config["ctx"], name="ctx"),
                conf_pattern(config["dat"], name="dat"),
                conf_pattern(config["sub"], name="sub"),
                conf_pattern(config["scf"], name="scf"),
                conf_pattern(config["aux"], name="aux"),
                conf_pattern(config["div"], name="div"),
                meth,
                conf_pattern(config["meth"][meth], name="run"),
            ),
            **conf_values(config["ctx"], name="ctx"),
            **conf_values(config["dat"], name="dat"),
            **conf_values(config["sub"], name="sub"),
            **conf_values(config["scf"], name="scf"),
            **conf_values(config["aux"], name="aux"),
            **conf_values(config["div"], name="div"),
            **conf_values(config["meth"][meth], name="run"),
        ),
        config["meth"],
    ),
)


rule plot:
    input:
        "sum/sim_disc_true.csv",
        "sum/sim_disc_resp.csv",
        "sum/.sim_disc_extra.flag",
    output:
        "sum/sim_disc_true_overview.pdf",
        "sum/sim_disc_true_n_vars.pdf",
        "sum/sim_disc_true_int_f.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_sim_disc.R"


rule summarize_true:
    input:
        target_files(directories, ["metrics_disc_true.yaml"]),
    output:
        "sum/sim_disc_true.csv",
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


rule summarize_resp:
    input:
        target_files(
            directories,
            ["metrics_disc_train_resp.yaml", "metrics_disc_test_resp.yaml"],
        ),
    output:
        "sum/sim_disc_resp.csv",
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
            "metrics_disc_{phs}_resp.yaml",
        ),
    localrule: True
    script:
        "../scripts/summarize.py"


rule extra:
    input:
        target_files(
            directories,
            [
                "roc.pdf",
                "prc.pdf",
                "adj.pdf",
                "confusion.pdf",
            ],
        ),
    output:
        "sum/.sim_disc_extra.flag",
    localrule: True
    shell:
        "touch {output}"
