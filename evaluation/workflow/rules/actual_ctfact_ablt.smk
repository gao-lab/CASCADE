include: "utils.smk"


directories = expand(
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
    **conf_values(config["meth"]["cascade"], name="run"),
)


rule plot:
    input:
        "sum/.actual_ctfact_ablt_selfless.flag",
        "sum/actual_ctfact_ablt.csv",
    output:
        "sum/actual_ctfact_ablt_train_single.pdf",
        "sum/actual_ctfact_ablt_train_double.pdf",
        "sum/actual_ctfact_ablt_test_single.pdf",
        "sum/actual_ctfact_ablt_test_double.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_actual_ctfact_ablt.R"


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
        "sum/actual_ctfact_ablt.csv",
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


rule selfless:
    input:
        target_files(
            directories,
            [
                "metrics_ctfact_train_each_selfless.csv",
                "metrics_ctfact_test_each_selfless.csv",
            ],
        ),
    output:
        "sum/.actual_ctfact_ablt_selfless.flag",
    shell:
        "touch {output}"
