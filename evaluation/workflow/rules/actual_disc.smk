include: "utils.smk"


def switch_scf(meth):
    if meth == "cascade":
        return config["scf"]
    return "cpl"


def switch_aux(meth):
    if meth == "cascade":
        return config["aux"]
    return "nil"


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
            **conf_values(switch_scf(meth), name="scf"),
            **conf_values(switch_aux(meth), name="aux"),
            **conf_values(config["div"], name="div"),
            **conf_values(config["meth"][meth], name="run"),
        ),
        config["meth"],
    ),
)


rule plot:
    input:
        "sum/actual_disc_resp.csv",
    output:
        "sum/actual_disc_resp_train_dist.pdf",
        "sum/actual_disc_resp_train_dist_diff.pdf",
        "sum/actual_disc_resp_train_acc.pdf",
        "sum/actual_disc_resp_test_dist.pdf",
        "sum/actual_disc_resp_test_dist_diff.pdf",
        "sum/actual_disc_resp_test_acc.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_actual_disc.R"


rule summarize_resp:
    input:
        target_files(
            directories,
            ["metrics_disc_train_resp.yaml", "metrics_disc_test_resp.yaml"],
        ),
    output:
        "sum/actual_disc_resp.csv",
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
