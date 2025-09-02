from os.path import join


rule plot_disc_curves:
    input:
        true=join("dat", "{ctx}", "{dat}", "{sub}", "sub.gml.gz"),
        pred=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "discover.gml.gz",
        ),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        roc=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "roc.pdf",
        ),
        prc=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "prc.pdf",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "plot_disc_curves.log",
        ),
    shell:
        "python workflow/scripts/plot_disc_curves.py "
        "--true {input.true} "
        "--pred {input.pred} "
        "--scaffold {input.scaffold} "
        "--roc {output.roc} "
        "--prc {output.prc} "
        "&> {log}"


rule plot_disc_adj:
    input:
        pred=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "discover.gml.gz",
        ),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "adj.pdf",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "plot_disc_adj.log",
        ),
    params:
        cluster=lambda wildcards: "--cluster" if wildcards.ctx.startswith("ds") else "",
    shell:
        "python workflow/scripts/plot_disc_adj.py "
        "--input {input.pred} "
        "--scaffold {input.scaffold} "
        "--output {output} "
        "{params.cluster} "
        "&> {log}"


rule plot_disc_confusion:
    input:
        true=join("dat", "{ctx}", "{dat}", "{sub}", "sub.gml.gz"),
        pred=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "discover.gml.gz",
        ),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "confusion.pdf",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "plot_disc_confusion.log",
        ),
    params:
        cutoff=lambda wildcards: (
            1e-5 if wildcards.meth in ("nobears", "notears", "dagma") else 0.5
        ),
    shell:
        "python workflow/scripts/plot_disc_confusion.py "
        "--true {input.true} "
        "--pred {input.pred} "
        "--scaffold {input.scaffold} "
        "--output {output} "
        "--cutoff {params.cutoff} "
        "&> {log}"


rule plot_true_adj:
    input:
        join("{path}", "{graph}.gml.gz"),
    output:
        join("{path}", "{graph}.pdf"),
    log:
        join("{path}", "plot_true_adj_{graph}.log"),
    wildcard_constraints:
        graph=r"causal|sub",
    shell:
        "python workflow/scripts/plot_adj.py "
        "--input {input} "
        "--output {output} "
        "&> {log}"


rule plot_dsgn_curve:
    input:
        lambda wildcards: [
            join(
                "inf",
                wildcards.ctx,
                wildcards.dat,
                wildcards.sub,
                wildcards.scf,
                wildcards.aux,
                wildcards.div,
                wildcards.meth,
                wildcards.run,
                f"dsgn_{wildcards.phs}",
                item,
                "dsgn.csv",
            )
            for item in listdir(checkpoints.request_designs.get(**wildcards).output[0])
            if not item.startswith(".")
            and not exists(
                join(
                    "inf",
                    wildcards.ctx,
                    wildcards.dat,
                    wildcards.sub,
                    wildcards.scf,
                    wildcards.aux,
                    wildcards.div,
                    wildcards.meth,
                    wildcards.run,
                    f"dsgn_{wildcards.phs}",
                    item,
                    ".blacklist",
                )
            )
        ],
    output:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "hrc_{phs}.pdf",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "plot_dsgn_curve_{phs}.log",
        ),
    shell:
        "python -u workflow/scripts/plot_dsgn_curve.py "
        "--designs {input} "
        "--output {output} "
        "&> {log}"
