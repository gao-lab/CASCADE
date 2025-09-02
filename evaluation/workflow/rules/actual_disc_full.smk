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


rule actual_disc_full:
    input:
        target_files(directories, ["tune.pt"]),
    output:
        "sum/.actual_disc_full.flag",
    localrule: True
    shell:
        "touch {output}"
