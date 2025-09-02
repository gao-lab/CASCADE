include: "utils.smk"


rule plot:
    input:
        "sum/actual_disc_hparams_resp.csv",
    output:
        "sum/actual_disc_hparams_resp_train_dist.pdf",
        "sum/actual_disc_hparams_resp_train_dist_diff.pdf",
        "sum/actual_disc_hparams_resp_train_acc.pdf",
        "sum/actual_disc_hparams_resp_test_dist.pdf",
        "sum/actual_disc_hparams_resp_test_dist_diff.pdf",
        "sum/actual_disc_hparams_resp_test_acc.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_actual_disc_hparams.R"


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
            ["metrics_disc_train_resp.yaml", "metrics_disc_test_resp.yaml"],
        ),
    output:
        "sum/actual_disc_hparams_resp.csv",
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
            "metrics_disc_{phs}_resp.yaml",
        ),
    localrule: True
    script:
        "../scripts/summarize.py"
