from os.path import join


include: "cascade.smk"


# ------------------------- Causal discovery methods ---------------------------


PC_RUN_CFG = "alpha={alpha}"


rule run_pc:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "pc",
            PC_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "pc",
            PC_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "pc",
            PC_RUN_CFG,
            "run_pc.log",
        ),
    threads: 8
    params:
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "pc",
            PC_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_pc.py "
        "--input {input.data} "
        "--scaffold {input.scaffold} "
        "--output {output.graph} "
        "--alpha {wildcards.alpha} "
        "--n-jobs {threads} "
        "--info {output.info} "
        "--verbose "
        "&> {log} || touch {params.blacklist}"


GES_RUN_CFG = "score={score}"


rule run_ges:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "ges",
            GES_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "ges",
            GES_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "ges",
            GES_RUN_CFG,
            "run_ges.log",
        ),
    params:
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "ges",
            GES_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_ges.py "
        "--input {input.data} "
        "--scaffold {input.scaffold} "
        "--output {output.graph} "
        "--score {wildcards.score} "
        "--info {output.info} "
        "--verbose "
        "&> {log} || touch {params.blacklist}"


GIES_RUN_CFG = "score={score}"


rule run_gies:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "gies",
            GIES_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "gies",
            GIES_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "gies",
            GIES_RUN_CFG,
            "run_gies.log",
        ),
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "gies",
            GIES_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_gies.py "
        "--input {input.data} "
        "--scaffold {input.scaffold} "
        "--output {output.graph} "
        "--interv-key {params.interv} "
        "--score {wildcards.score} "
        "--info {output.info} "
        "--verbose "
        "&> {log} || touch {params.blacklist}"


rule run_gsp:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "gsp",
            "default",
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "gsp",
            "default",
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "gsp",
            "default",
            "run_gsp.log",
        ),
    threads: 8
    singularity:
        "workflow/envs/causaldag.sif"
    params:
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "gsp",
            "default",
            ".blacklist",
        ),
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_gsp.py "
        "--input {input.data} "
        "--scaffold {input.scaffold} "
        "--output {output.graph} "
        "--n-jobs {threads} "
        "--info {output.info} "
        "--verbose "
        "&> {log} || touch {params.blacklist}"


rule run_igsp:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "igsp",
            "default",
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "igsp",
            "default",
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "igsp",
            "default",
            "run_igsp.log",
        ),
    threads: 8
    singularity:
        "workflow/envs/causaldag.sif"
    params:
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "igsp",
            "default",
            ".blacklist",
        ),
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_igsp.py "
        "--input {input.data} "
        "--scaffold {input.scaffold} "
        "--output {output.graph} "
        "--n-jobs {threads} "
        "--info {output.info} "
        "--verbose "
        "&> {log} || touch {params.blacklist}"


ICP_RUN_CFG = "deg_limit={deg_limit}-alpha={alpha}"


rule run_icp:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        scaffold=join("dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "icp",
            ICP_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "icp",
            ICP_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "icp",
            ICP_RUN_CFG,
            "run_icp.log",
        ),
    threads: 8
    singularity:
        "workflow/envs/causalicp.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "nil",
            "{div}",
            "icp",
            ICP_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_icp.py "
        "--input {input.data} "
        "--scaffold {input.scaffold} "
        "--output {output.graph} "
        "--interv-key {params.interv} "
        "--deg-limit {wildcards.deg_limit} "
        "--alpha {wildcards.alpha} "
        "--n-jobs {threads} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


NOTEARS_RUN_CFG = "hid={hid}-lam1={lam1}-lam2={lam2}-run_sd={run_sd}"


rule run_notears:
    input:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "notears",
            NOTEARS_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "notears",
            NOTEARS_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "notears",
            NOTEARS_RUN_CFG,
            "run_notears.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/notears.sif"
    params:
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "notears",
            NOTEARS_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_notears.py "
        "--input {input} "
        "--output {output.graph} "
        "--hid-dim {wildcards.hid} "
        "--lambda1 {wildcards.lam1} "
        "--lambda2 {wildcards.lam2} "
        "--random-seed {wildcards.run_sd} "
        "--info {output.info} "
        "--gpu "
        "&> {log} || touch {params.blacklist}"


NOBEARS_RUN_CFG = "poly={poly}-rho={rho}-run_sd={run_sd}"


rule run_nobears:
    input:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train_downgrade.h5ad"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "nobears",
            NOBEARS_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "nobears",
            NOBEARS_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "nobears",
            NOBEARS_RUN_CFG,
            "run_nobears.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/nobears.sif"
    params:
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "nobears",
            NOBEARS_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_nobears.py "
        "--input {input} "
        "--output {output.graph} "
        "--poly-degree {wildcards.poly} "
        "--rho-init {wildcards.rho} "
        "--random-seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


DCDI_RUN_CFG = "layer={layer}-hid={hid}-nonlin={nonlin}-reg={reg}-run_sd={run_sd}"


rule run_dcdi:
    input:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dcdi",
            DCDI_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dcdi",
            DCDI_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dcdi",
            DCDI_RUN_CFG,
            "run_dcdi.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/dcdi.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dcdi",
            DCDI_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_dcdi.py "
        "--input {input} "
        "--output {output.graph} "
        "--interv-key {params.interv} "
        "--info {output.info} "
        "--train "
        "--random-seed {wildcards.run_sd} "
        "--model DCDI-DSF "
        "--flow-num-layers {wildcards.layer} "
        "--flow-hid-dim {wildcards.hid} "
        "--nonlin {wildcards.nonlin} "
        "--intervention "
        "--intervention-type perfect "
        "--intervention-knowledge known "
        "--reg-coeff {wildcards.reg} "
        "--no-w-adjs-log "
        "--gpu "
        "&> {log} || touch {params.blacklist}"


DCDFG_RUN_CFG = (
    "n_mod={n_mod}-reg={reg}-cstr={cstr}-model={model}-poly={poly}-run_sd={run_sd}"
)


rule run_dcdfg:
    input:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dcdfg",
            DCDFG_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dcdfg",
            DCDFG_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dcdfg",
            DCDFG_RUN_CFG,
            "run_dcdfg.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/dcdfg.sif"
    params:
        interv=get_interv,
        poly=lambda wildcards: "--poly" if wildcards.poly == "True" else "",
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dcdfg",
            DCDFG_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_dcdfg.py "
        "--input {input} "
        "--output {output.graph} "
        "--interv-key {params.interv} "
        "--random-seed {wildcards.run_sd} "
        "--info {output.info} "
        "--num-modules {wildcards.n_mod} "
        "--reg-coeff {wildcards.reg} "
        "--constraint-mode {wildcards.cstr} "
        "--model {wildcards.model} "
        "{params.poly} "
        "&> {log} || touch {params.blacklist}"


DAGMA_RUN_CFG = "hid={hid}-lam1={lam1}-lam2={lam2}-T={T}-run_sd={run_sd}"


rule run_dagma:
    input:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dagma",
            DAGMA_RUN_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dagma",
            DAGMA_RUN_CFG,
            "info.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dagma",
            DAGMA_RUN_CFG,
            "run_dagma.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/dagma.sif"
    params:
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "dagma",
            DAGMA_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_dagma.py "
        "--input {input} "
        "--output {output.graph} "
        "--hid-dim {wildcards.hid} "
        "--lambda1 {wildcards.lam1} "
        "--lambda2 {wildcards.lam2} "
        "-T {wildcards.T} "
        "--random-seed {wildcards.run_sd} "
        "--info {output.info} "
        "--gpu "
        "--verbose "
        "&> {log} || touch {params.blacklist}"


# -------------------------- Counterfactual methods ----------------------------


rule run_no_change:
    input:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
    output:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "no_change",
            "run",
            "ctfact_{phs}.h5ad",
        ),
    localrule: True
    shell:
        "ln -fnrs {input} {output}"


rule run_mean:
    input:
        train=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "mean",
            "run",
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "mean",
            "run",
            "info_{phs}.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "mean",
            "run",
            "run_mean_{phs}.log",
        ),
    params:
        interv=get_interv,
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_mean.py "
        "--input-train {input.train} "
        "--input-data {input.data} "
        "--output {output.data} "
        "--interv-key {params.interv} "
        "--info {output.info} "
        "&> {log}"


rule run_additive_predict:
    input:
        train=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "additive",
            "run",
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "additive",
            "run",
            "info_{phs}.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "additive",
            "run",
            "run_additive_{phs}.log",
        ),
    params:
        interv=get_interv,
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_additive.py predict "
        "--input-train {input.train} "
        "--input-data {input.data} "
        "--output {output.data} "
        "--interv-key {params.interv} "
        "--info {output.info} "
        "&> {log}"


rule run_additive_design:
    input:
        train=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        target=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "{name}.h5ad"
        ),
        pool=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "pool.txt"),
    output:
        design=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "additive",
            "run",
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "additive",
            "run",
            "dsgn_{phs}",
            "{name}",
            "info_design.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "additive",
            "run",
            "dsgn_{phs}",
            "{name}",
            "run_additive_design.log",
        ),
    params:
        interv=get_interv,
        size=lambda wildcards: len(wildcards.name.split("+")),
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_additive.py design "
        "--input-train {input.train} "
        "--input-target {input.target} "
        "--pool {input.pool} "
        "--output {output.design} "
        "--interv-key {params.interv} "
        "--design-size {params.size} "
        "--info {output.info} "
        "&> {log}"


LINEAR_RUN_CFG = "dim={dim}-lam={lam}-run_sd={run_sd}"


rule run_linear_train:
    input:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
    output:
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "model.npz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "info_train.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "run_linear_train.log",
        ),
    threads: 8
    params:
        interv=get_interv,
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_linear.py train "
        "--input {input} "
        "--output {output.model} "
        "--interv-key {params.interv} "
        "--dim {wildcards.dim} "
        "--lam {wildcards.lam} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log}"


rule run_linear_predict:
    input:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "ctfact_{phs}.h5ad",
        ),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "model.npz",
        ),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "info_predict_{phs}.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "run_linear_predict_{phs}.log",
        ),
    threads: 8
    params:
        interv=get_interv,
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_linear.py predict "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--output {output.data} "
        "--interv-key {params.interv} "
        "--info {output.info} "
        "&> {log}"


rule run_linear_design:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "model.npz",
        ),
        target=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "{name}.h5ad"
        ),
        pool=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "pool.txt"),
    output:
        design=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "info_design.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "nil",
            "{div}",
            "linear",
            LINEAR_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "run_linear_design.log",
        ),
    params:
        size=lambda wildcards: len(wildcards.name.split("+")),
    shell:
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_linear.py design "
        "--input-model {input.model} "
        "--input-target {input.target} "
        "--pool {input.pool} "
        "--output {output.design} "
        "--design-size {params.size} "
        "--info {output.info} "
        "&> {log}"


GEARS_RUN_CFG = "hidden_size={hidden_size}-epochs={epochs}-run_sd={run_sd}"


rule run_gears_train:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        go_path=join("dat", "go_files"),
    output:
        model=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "cpl",
                "go",
                "{div}",
                "gears",
                GEARS_RUN_CFG,
                "model",
            )
        ),
        data_path=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "cpl",
                "go",
                "{div}",
                "gears",
                GEARS_RUN_CFG,
                "data_path",
            )
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "info_train.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "run_gears_train.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/gears.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_gears.py train "
        "--input {input.data} "
        "--go-path {input.go_path} "
        "--output {output.model} "
        "--data-path {output.data_path} "
        "--interv-key {params.interv} "
        "--hidden-size {wildcards.hidden_size} "
        "--epochs {wildcards.epochs} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_gears_predict:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "model",
        ),
        data_path=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "data_path",
        ),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "info_predict_{phs}.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "run_gears_predict_{phs}.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/gears.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_gears.py predict "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--output {output.data} "
        "--data-path {input.data_path} "
        "--interv-key {params.interv} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_gears_design:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "model",
        ),
        target=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "{name}.h5ad"
        ),
        pool=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "pool.txt"),
        data_path=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "data_path",
        ),
    output:
        design=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "info_design.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "run_gears_design.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/gears.sif"
    params:
        size=lambda wildcards: len(wildcards.name.split("+")),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "gears",
            GEARS_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_gears.py design "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--input-target {input.target} "
        "--pool {input.pool} "
        "--output {output.design} "
        "--data-path {input.data_path} "
        "--design-size {params.size} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


CPA_RUN_CFG = "run_sd={run_sd}"


rule run_cpa_train:
    input:
        data_train=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        data_test=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_test.h5ad"),
        pert_emb=join("dat", "gene2gos_lsi.csv.gz"),
    output:
        model=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "cpl",
                "lsi",
                "{div}",
                "cpa",
                CPA_RUN_CFG,
                "model",
            )
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "info_train.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "run_cpa_train.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/cpa.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_cpa.py train "
        "--input-train {input.data_train} "
        "--input-test {input.data_test} "
        "--pert-emb {input.pert_emb} "
        "--output {output.model} "
        "--interv-key {params.interv} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_cpa_predict:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "model",
        ),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "info_predict_{phs}.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "run_cpa_predict_{phs}.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/cpa.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_cpa.py predict "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--output {output.data} "
        "--interv-key {params.interv} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_cpa_design:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "model",
        ),
        target=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "{name}.h5ad"
        ),
        pool=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "pool.txt"),
    output:
        design=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "info_design.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "run_cpa_design.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/cpa.sif"
    params:
        size=lambda wildcards: len(wildcards.name.split("+")),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "cpa",
            CPA_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_cpa.py design "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--input-target {input.target} "
        "--pool {input.pool} "
        "--output {output.design} "
        "--design-size {params.size} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


BIOLORD_RUN_CFG = "run_sd={run_sd}"


rule run_biolord_train:
    input:
        data_train=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        data_test=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_test.h5ad"),
        pert_emb=join("dat", "gene2gos_lsi.csv.gz"),
    output:
        model=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "cpl",
                "lsi",
                "{div}",
                "biolord",
                BIOLORD_RUN_CFG,
                "model",
            )
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "info_train.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "run_biolord_train.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/biolord.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_biolord.py train "
        "--input-train {input.data_train} "
        "--input-test {input.data_test} "
        "--pert-emb {input.pert_emb} "
        "--output {output.model} "
        "--interv-key {params.interv} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_biolord_predict:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "model",
        ),
        pert_emb=join("dat", "gene2gos_lsi.csv.gz"),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "info_predict_{phs}.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "run_biolord_predict_{phs}.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/biolord.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_biolord.py predict "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--pert-emb {input.pert_emb} "
        "--output {output.data} "
        "--interv-key {params.interv} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_biolord_design:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "model",
        ),
        target=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "{name}.h5ad"
        ),
        pert_emb=join("dat", "gene2gos_lsi.csv.gz"),
        pool=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "pool.txt"),
    output:
        design=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "info_design.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "run_biolord_design.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/biolord.sif"
    params:
        size=lambda wildcards: len(wildcards.name.split("+")),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "lsi",
            "{div}",
            "biolord",
            BIOLORD_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_biolord.py design "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--input-target {input.target} "
        "--pert-emb {input.pert_emb} "
        "--pool {input.pool} "
        "--output {output.design} "
        "--design-size {params.size} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


SCGPT_RUN_CFG = "epochs={epochs}-run_sd={run_sd}"


rule run_scgpt_train:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        go_path=join("dat", "go_files"),
    output:
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "model",
            "model.pt",
        ),
        data_path=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "cpl",
                "go",
                "{div}",
                "scgpt",
                SCGPT_RUN_CFG,
                "data_path",
            )
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "info_train.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "run_scgpt_train.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/scgpt.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_scgpt.py train "
        "--input {input.data} "
        "--go-path {input.go_path} "
        "--output {output.model} "
        "--data-path {output.data_path} "
        "--interv-key {params.interv} "
        "--epochs {wildcards.epochs} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_scgpt_predict:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "model",
            "model.pt",
        ),
        data_path=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "data_path",
        ),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "info_predict_{phs}.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "run_scgpt_predict_{phs}.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/scgpt.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_scgpt.py predict "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--output {output.data} "
        "--data-path {input.data_path} "
        "--interv-key {params.interv} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_scgpt_design:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "model",
            "model.pt",
        ),
        target=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "{name}.h5ad"
        ),
        pool=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "pool.txt"),
        data_path=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "data_path",
        ),
    output:
        design=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "info_design.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "run_scgpt_design.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/scgpt.sif"
    params:
        size=lambda wildcards: len(wildcards.name.split("+")),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scgpt",
            SCGPT_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_scgpt.py design "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--input-target {input.target} "
        "--pool {input.pool} "
        "--output {output.design} "
        "--data-path {input.data_path} "
        "--design-size {params.size} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


SCFOUNDATION_RUN_CFG = "hidden_size={hidden_size}-epochs={epochs}-run_sd={run_sd}"


rule run_scfoundation_train:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        go_path=join("dat", "go_files"),
    output:
        model=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "cpl",
                "go",
                "{div}",
                "scfoundation",
                SCFOUNDATION_RUN_CFG,
                "model",
            )
        ),
        data_path=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "cpl",
                "go",
                "{div}",
                "scfoundation",
                SCFOUNDATION_RUN_CFG,
                "data_path",
            )
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "info_train.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "run_scfoundation_train.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/scfoundation.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "micromamba run python workflow/scripts/run_scfoundation.py train "
        "--input {input.data} "
        "--go-path {input.go_path} "
        "--output {output.model} "
        "--data-path {output.data_path} "
        "--interv-key {params.interv} "
        "--hidden-size {wildcards.hidden_size} "
        "--epochs {wildcards.epochs} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_scfoundation_predict:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "model",
        ),
        data_path=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "data_path",
        ),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "info_predict_{phs}.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "run_scfoundation_predict_{phs}.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/scfoundation.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "micromamba run python workflow/scripts/run_scfoundation.py predict "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--output {output.data} "
        "--data-path {input.data_path} "
        "--interv-key {params.interv} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "--hidden-size {wildcards.hidden_size} "
        "&> {log} || touch {params.blacklist}"


rule run_scfoundation_design:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "model",
        ),
        target=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "{name}.h5ad"
        ),
        pool=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "pool.txt"),
        data_path=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "data_path",
        ),
    output:
        design=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "info_design.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "run_scfoundation_design.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/scfoundation.sif"
    params:
        size=lambda wildcards: len(wildcards.name.split("+")),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "go",
            "{div}",
            "scfoundation",
            SCFOUNDATION_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "micromamba run python workflow/scripts/run_scfoundation.py design "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--input-target {input.target} "
        "--pool {input.pool} "
        "--output {output.design} "
        "--data-path {input.data_path} "
        "--design-size {params.size} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "--hidden-size {wildcards.hidden_size} "
        "&> {log} || touch {params.blacklist}"


STATE_RUN_CFG = "cell_set_len={cell_set_len}-hidden_dim={hidden_dim}-run_sd={run_sd}"


rule run_state_train:
    input:
        data_train=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        data_test=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "test.h5ad"),
    output:
        model=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "cpl",
                "esm",
                "{div}",
                "state",
                STATE_RUN_CFG,
                "model",
            )
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "info_train.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "run_state_train.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/state.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "python workflow/scripts/run_state.py train "
        "--input-train {input.data_train} "
        "--input-test {input.data_test} "
        "--output {output.model} "
        "--interv-key {params.interv} "
        "--cell-set-len {wildcards.cell_set_len} "
        "--hidden-dim {wildcards.hidden_dim} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


rule run_state_predict:  # Only supports test
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_test.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "model",
        ),
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "ctfact_test.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "info_predict_test.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "run_state_predict_test.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/state.sif"
    params:
        interv=get_interv,
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "python workflow/scripts/run_state.py predict "
        "--data {input.data} "
        "--model {input.model} "
        "--output {output.data} "
        "--interv-key {params.interv} "
        "--info {output.info} "
        "&> {log} || touch {params.blacklist}"


# TODO
rule run_state_design:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "model",
        ),
        target=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "{name}.h5ad"
        ),
        pool=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "dsgn_{phs}", "pool.txt"),
    output:
        design=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "info_design.yaml",
        ),
    log:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            "run_state_design.log",
        ),
    threads: 8
    resources:
        gpu=1,
    singularity:
        "workflow/envs/state.sif"
    params:
        size=lambda wildcards: len(wildcards.name.split("+")),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "cpl",
            "esm",
            "{div}",
            "state",
            STATE_RUN_CFG,
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[gpu_prefix]}"
        "{config[cmd_prefix]}"
        "python workflow/scripts/run_state.py design "
        "--input-data {input.data} "
        "--input-model {input.model} "
        "--input-target {input.target} "
        "--pool {input.pool} "
        "--output {output.design} "
        "--data-path {input.data_path} "
        "--design-size {params.size} "
        "--seed {wildcards.run_sd} "
        "--info {output.info} "
        "--hidden-size {wildcards.hidden_size} "
        "&> {log} || touch {params.blacklist}"
