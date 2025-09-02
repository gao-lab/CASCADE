from os import listdir
from os.path import join
import anndata as ad


def get_log_normalize(wildcards):
    if "NegBin" in wildcards.run:
        return "--log-normalize"
    return ""


def get_resp(wildcards):
    if wildcards.phs == "train":
        return join(
            "dat",
            wildcards.ctx,
            "imp=0" if "imp" in wildcards.dat else wildcards.dat,
            wildcards.sub,
            wildcards.div,
            "train.h5ad",
        )
    # wildcards.phs == "test"
    return [
        join(
            "dat",
            wildcards.ctx,
            "imp=0" if "imp" in wildcards.dat else wildcards.dat,
            wildcards.sub,
            "ctrl.h5ad",
        ),
        join(
            "dat",
            wildcards.ctx,
            "imp=0" if "imp" in wildcards.dat else wildcards.dat,
            wildcards.sub,
            wildcards.div,
            "test.h5ad",
        ),
    ]


rule metrics_disc_true:
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
            "metrics_disc_true.yaml",
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
            "metrics_disc_true.log",
        ),
    params:
        cutoff=lambda wildcards: (
            1e-5 if wildcards.meth in ("nobears", "notears", "dagma") else 0.5
        ),
    shell:
        "python -u workflow/scripts/metrics_disc.py true "
        "--true {input.true} "
        "--pred {input.pred} "
        "--scaffold {input.scaffold} "
        "--output {output} "
        "--cutoff {params.cutoff} "
        "&> {log}"


rule metrics_disc_resp:
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
        resp=get_resp,
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
            "metrics_disc_{phs}_resp.yaml",
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
            "metrics_disc_{phs}_resp.log",
        ),
    params:
        cutoff=lambda wildcards: (
            1e-5 if wildcards.meth in ("nobears", "notears", "dagma") else 0.5
        ),
    shell:
        "python -u workflow/scripts/metrics_disc.py resp "
        "--pred {input.pred} "
        "--resp {input.resp} "
        "--output {output} "
        "--cutoff {params.cutoff} "
        "&> {log}"


rule metrics_ctfact:
    input:
        ctrl=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        true=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "{phs}.h5ad"),
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
            "ctfact_{phs}.h5ad",
        ),
    output:
        each=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "metrics_ctfact_{phs}_each.csv",
        ),
        category=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "metrics_ctfact_{phs}_category.yaml",
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
            "metrics_ctfact_{phs}.log",
        ),
    params:
        log_normalize=get_log_normalize,
    threads: 8  # Avoid over-stressing individual nodes
    shell:
        "python -u workflow/scripts/metrics_ctfact.py "
        "--ctrl {input.ctrl} "
        "--true {input.true} "
        "--pred {input.pred} "
        "--output-each {output.each} "
        "--output-category {output.category} "
        "--top-de {config[top_de]} "
        "{params.log_normalize} "
        "&> {log}"


rule metrics_ctfact_selfless:
    input:
        ctrl=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        true=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "{phs}.h5ad"),
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
            "ctfact_{phs}.h5ad",
        ),
    output:
        each=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "metrics_ctfact_{phs}_each_selfless.csv",
        ),
        category=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "{meth}",
            "{run}",
            "metrics_ctfact_{phs}_category_selfless.yaml",
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
            "metrics_ctfact_{phs}_selfless.log",
        ),
    params:
        log_normalize=get_log_normalize,
    threads: 8  # Avoid over-stressing individual nodes
    shell:
        "python -u workflow/scripts/metrics_ctfact.py "
        "--ctrl {input.ctrl} "
        "--true {input.true} "
        "--pred {input.pred} "
        "--output-each {output.each} "
        "--output-category {output.category} "
        "--top-de {config[top_de]} "
        "--exclude-self "
        "{params.log_normalize} "
        "&> {log}"


rule metrics_dsgn:
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
            "metrics_dsgn_{phs}.yaml",
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
            "metrics_dsgn_{phs}.log",
        ),
    shell:
        "python -u workflow/scripts/metrics_dsgn.py "
        "--designs {input} "
        "--output {output} "
        "&> {log}"
