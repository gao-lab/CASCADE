include: "utils.smk"


def switch_scf(meth):
    if meth in (
        "no_change",
        "mean",
        "additive",
        "linear",
        "gears",
        "cpa",
        "biolord",
        "scgpt",
        "scfoundation",
        "state",
    ):
        return "cpl"
    return config["scf"]


def switch_aux(meth):
    if meth in ("gears", "scgpt", "scfoundation"):
        return "go"
    if meth in ("cpa", "biolord"):
        return "lsi"
    if meth in ("state"):
        return "esm"
    if meth in ("no_change", "mean", "additive", "linear"):
        return "nil"
    return config["aux"]


def switch_metric(directory):
    if "/state/" in directory:  # State cannot predict on training set
        return ["metrics_ctfact_test_category.yaml"]
    return [
        "metrics_ctfact_train_category.yaml",
        "metrics_ctfact_test_category.yaml",
    ]


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
        "sum/actual_ctfact.csv",
    output:
        "sum/actual_ctfact_train_single.pdf",
        "sum/actual_ctfact_train_double.pdf",
        "sum/actual_ctfact_test_single.pdf",
        "sum/actual_ctfact_test_double.pdf",
    localrule: True
    shell:
        "cd workflow/scripts && Rscript plot_actual_ctfact.R"


rule summarize:
    input:
        target_files(directories, switch_metric),
    output:
        "sum/actual_ctfact.csv",
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
            "metrics_ctfact_{phs}_category.yaml",
        ),
    localrule: True
    script:
        "../scripts/summarize.py"
