include: "utils.smk"


def switch_scf(meth):
    if meth in (
        "linear",
        "additive",
        "cpa",
        "biolord",
        "gears",
        "scgpt",
        "scfoundation",
    ):
        return "cpl"
    return config["scf"]


def switch_aux(meth):
    if meth in ("linear", "additive"):
        return "nil"
    if meth in ("gears", "scgpt", "scfoundation"):
        return "go"
    if meth in ("cpa", "biolord"):
        return "lsi"
    return config["aux"]


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
        "sum/actual_dsgn.csv",
        "sum/.actual_dsgn_extra.flag",
    output:
        "sum/actual_dsgn_train.pdf",
        "sum/actual_dsgn_test.pdf",
        "sum/actual_dsgn_train_partial.pdf",
        "sum/actual_dsgn_test_partial.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_actual_dsgn.R"


rule summarize:
    input:
        target_files(
            directories,
            ["metrics_dsgn_train.yaml", "metrics_dsgn_test.yaml"],
        ),
    output:
        "sum/actual_dsgn.csv",
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
            "metrics_dsgn_{phs}.yaml",
        ),
    localrule: True
    script:
        "../scripts/summarize.py"


rule extra:
    input:
        target_files(
            directories,
            ["hrc_train.pdf", "hrc_test.pdf"],
        ),
    output:
        "sum/.actual_dsgn_extra.flag",
    localrule: True
    shell:
        "touch {output}"
