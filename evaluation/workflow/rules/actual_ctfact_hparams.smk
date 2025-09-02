include: "utils.smk"


directories = expand(
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
)


rule plot:
    input:
        "sum/actual_ctfact_hparams.csv",
    output:
        "sum/actual_ctfact_hparams_train_delta_pcc_single.pdf",
        "sum/actual_ctfact_hparams_train_normalized_mse_single.pdf",
        "sum/actual_ctfact_hparams_train_delta_pcc_double.pdf",
        "sum/actual_ctfact_hparams_train_normalized_mse_double.pdf",
        "sum/actual_ctfact_hparams_test_delta_pcc_single.pdf",
        "sum/actual_ctfact_hparams_test_normalized_mse_single.pdf",
        "sum/actual_ctfact_hparams_test_delta_pcc_double.pdf",
        "sum/actual_ctfact_hparams_test_normalized_mse_double.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_actual_ctfact_hparams.R"


rule summarize:
    input:
        target_files(
            directories,
            [
                "metrics_ctfact_train_category.yaml",
                "metrics_ctfact_test_category.yaml",
            ],
        ),
    output:
        "sum/actual_ctfact_hparams.csv",
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
            "metrics_ctfact_{phs}_category.yaml",
        ),
    localrule: True
    script:
        "../scripts/summarize.py"
