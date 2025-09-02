include: "utils.smk"


rule plot:
    input:
        "sum/actual_dsgn_hparams.csv",
    output:
        "sum/actual_dsgn_hparams_train_single.pdf",
        "sum/actual_dsgn_hparams_train_double.pdf",
        "sum/actual_dsgn_hparams_test_single.pdf",
        "sum/actual_dsgn_hparams_test_double.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_actual_dsgn_hparams.R"


rule summarize:
    input:
        target_files(
            expand(
                expand(
                    join(
                        "inf",
                        conf_pattern(config["ctx"], name="ctx"),
                        conf_pattern(config["dat"], name="dat"),
                        conf_pattern(config["sub"], name="sub"),
                        conf_pattern(config["scf"], name="scf"),
                        conf_pattern(config["aux"], name="aux"),
                        conf_pattern(config["div"], name="div"),
                        "cascade",
                        conf_pattern(config["meth"]["cascade"]),
                    ),
                    **conf_values(config["ctx"], name="ctx"),
                    **conf_values(config["dat"], name="dat"),
                    **conf_values(config["sub"], name="sub"),
                    **conf_values(config["scf"], name="scf"),
                    **conf_values(config["aux"], name="aux"),
                    **conf_values(config["div"], name="div"),
                    allow_missing=True,
                ),
                zip,
                **conf_values_zip(config["meth"]["cascade"]),
            ),
            ["metrics_dsgn_train_category.yaml", "metrics_dsgn_test_category.yaml"],
        ),
    output:
        "sum/actual_dsgn_hparams.csv",
    params:
        pattern=lambda wildcards: join(
            "inf",
            conf_pattern(config["ctx"], name="ctx"),
            conf_pattern(config["dat"], name="dat"),
            conf_pattern(config["sub"], name="sub"),
            conf_pattern(config["scf"], name="scf"),
            conf_pattern(config["aux"], name="aux"),
            conf_pattern(config["div"], name="div"),
            "cascade",
            conf_pattern(config["meth"]["cascade"]),
            "metrics_dsgn_{phs}_category.yaml",
        ),
    localrule: True
    script:
        "../scripts/summarize.py"
