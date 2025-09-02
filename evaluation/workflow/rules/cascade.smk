from os.path import join, exists


def get_interv(wildcards):
    if wildcards.ctx.startswith("ds"):
        ds = wildcards.ctx.split("=")[1]
        return config["datasets"][ds]["interv"]
    return "knockout"


def get_latent_mod(wildcards):
    if wildcards.aux == "nil" or wildcards.dz == "0":
        return "NilLatent"
    if wildcards.aux in ("svd", "scgpt", "lsi"):
        return "EmbLatent"
    if wildcards.aux == "go":
        return "GCNLatent"
    raise ValueError(f"Unknown aux: {wildcards.aux}")


def get_use_covariate(wildcards):
    if wildcards.ctx.startswith("ds"):
        ds = wildcards.ctx.split("=")[1]
        return f"--use-covariate {config['datasets'][ds]['covariate']}"
    return ""


def get_use_size(wildcards):
    if wildcards.lik == "NegBin":
        return "--use-size ncounts"
    return ""


def get_use_layer(wildcards):
    if wildcards.lik == "NegBin":
        return "--use-layer counts"
    return ""


def get_latent_data(wildcards, input):
    if input.latent_data:
        return f"--latent-data {input.latent_data}"
    return ""


def get_tune_ct(wildcards):
    if wildcards.tune_ct == "True":
        return "--tune-ctfact"
    return ""


def get_ablation(wildcards):
    if wildcards.ablt == "latent":
        return "--ablate-latent"
    if wildcards.ablt == "interv":
        return "--ablate-interv"
    if wildcards.ablt == "graph":
        return "--ablate-graph"
    return ""


def get_design_size(wildcards):
    return len(wildcards.name.split("+"))


def get_bs(bs):
    def wrapped(wildcards, input, output, threads, resources):
        return round(bs / resources.gpu)

    return wrapped


def latent_data_switch(wildcards):
    if wildcards.aux == "nil":
        return []
    if wildcards.aux in ("svd", "scgpt", "lsi"):
        return join(
            "dat",
            wildcards.ctx,
            wildcards.dat,
            wildcards.sub,
            wildcards.aux,
            "latent.csv.gz",
        )
    if wildcards.aux == "go":
        return join(
            "dat",
            wildcards.ctx,
            wildcards.dat,
            wildcards.sub,
            wildcards.aux,
            "latent.gml.gz",
        )
    raise ValueError(f"Unknown aux: {wildcards.aux}")


def discover_model_switch(wildcards):
    dir = join(
        "inf",
        wildcards.ctx,
        wildcards.dat,
        wildcards.sub,
        wildcards.scf,
        wildcards.aux,
        wildcards.div,
        "cascade",
        CASCADE_DISC_CFG.format(**wildcards),
    )
    if exists(join(dir, ".blacklist")):
        return join(dir, ".blacklist")
    return join(dir, "discover.pt")


def discover_graph_switch(wildcards):
    dir = join(
        "inf",
        wildcards.ctx,
        wildcards.dat,
        wildcards.sub,
        wildcards.scf,
        wildcards.aux,
        wildcards.div,
        "cascade",
        CASCADE_DISC_CFG.format(**wildcards),
    )
    if exists(join(dir, ".blacklist")):
        return join(dir, ".blacklist")
    return join(dir, "discover.gml.gz")


def tune_model_switch(wildcards):
    dir = join(
        "inf",
        wildcards.ctx,
        wildcards.dat,
        wildcards.sub,
        wildcards.scf,
        wildcards.aux,
        wildcards.div,
        "cascade",
        CASCADE_TUNE_CFG.format(**wildcards),
    )
    if exists(join(dir, ".blacklist")):
        return join(dir, ".blacklist")
    return join(dir, "tune.pt")


CASCADE_DISC_CFG = (
    "nptc={nptc}-"
    "dz={dz}-"
    "beta={beta}-"
    "sps={sps}-"
    "acyc={acyc}-"
    "lik={lik}-"
    "lam={lam}-"
    "alp={alp}-"
    "run_sd={run_sd}"
)
CASCADE_TUNE_CFG = CASCADE_DISC_CFG + "-tune_ct={tune_ct}"
CASCADE_CTFACT_CFG = CASCADE_TUNE_CFG + "-ablt={ablt}"
CASCADE_DSGN_CFG = CASCADE_TUNE_CFG + "-dsgn={dsgn}"


rule cascade_disc:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        scaffold_graph=join(
            "dat", "{ctx}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"
        ),
        latent_data=latent_data_switch,
    output:
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DISC_CFG,
            "discover.pt",
        ),
        log_dir=directory(
            join(
                "inf",
                "{ctx}",
                "{dat}",
                "{sub}",
                "{scf}",
                "{aux}",
                "{div}",
                "cascade",
                CASCADE_DISC_CFG,
                "log_dir",
            )
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DISC_CFG,
            "info_disc.yaml",
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
            "cascade",
            CASCADE_DISC_CFG,
            "cascade_disc.log",
        ),
    threads: 8
    resources:
        gpu=1,
    params:
        interv=get_interv,
        lat=get_latent_mod,
        use_covariate=get_use_covariate,
        use_size=get_use_size,
        use_layer=get_use_layer,
        latent_data=get_latent_data,
        bs=get_bs(128),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DISC_CFG,
            ".blacklist",
        ),
    shell:
        "{config[cascade_prefix]}"
        "{config[cmd_prefix]}"
        "cascade discover "
        "-d {input.data} "
        "-m {output.model} "
        "-i {output.info} "
        "--interv-key {params.interv} "
        "{params.use_covariate} "
        "{params.use_size} "
        "{params.use_layer} "
        "--n-particles {wildcards.nptc} "
        "--latent-dim {wildcards.dz} "
        "--beta {wildcards.beta} "
        "--sparse-mod {wildcards.sps} "
        "--acyc-mod {wildcards.acyc} "
        "--latent-mod {params.lat} "
        "--lik-mod {wildcards.lik} "
        "--scaffold-graph {input.scaffold_graph} "
        "{params.latent_data} "
        "--lam {wildcards.lam} "
        "--alpha {wildcards.alp} "
        "--random-seed {wildcards.run_sd} "
        "--log-dir {output.log_dir} "
        "--batch-size {params.bs} "
        "--n-devices {resources.gpu} "
        "--random-sleep 60 "
        "-v "
        "&> {log} || touch {params.blacklist}"


rule cascade_acyc:
    input:
        join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DISC_CFG,
            "discover.pt",
        ),
    output:
        graph=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DISC_CFG,
            "discover.gml.gz",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DISC_CFG,
            "info_acyc.yaml",
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
            "cascade",
            CASCADE_DISC_CFG,
            "cascade_acyc.log",
        ),
    threads: lambda wildcards: int(wildcards.nptc)
    params:
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DISC_CFG,
            ".blacklist",
        ),
    shell:
        "{config[cascade_prefix]}"
        "{config[cmd_prefix]}"
        "cascade acyclify "
        "-m {input} "
        "-g {output.graph} "
        "-i {output.info} "
        "&> {log} || touch {params.blacklist}"


rule cascade_tune:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "train.h5ad"),
        graph=discover_graph_switch,
        model=discover_model_switch,
    output:
        model=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_TUNE_CFG,
            "tune.pt",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_TUNE_CFG,
            "info_tune.yaml",
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
            "cascade",
            CASCADE_TUNE_CFG,
            "cascade_tune.log",
        ),
    threads: 8
    resources:
        gpu=1,
    params:
        interv=get_interv,
        use_covariate=get_use_covariate,
        use_size=get_use_size,
        use_layer=get_use_layer,
        tune_ct=get_tune_ct,
        bs=get_bs(128),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_TUNE_CFG,
            ".blacklist",
        ),
    shell:
        "{config[cascade_prefix]}"
        "{config[cmd_prefix]}"
        "cascade tune "
        "-d {input.data} "
        "-g {input.graph} "
        "-m {input.model} "
        "-o {output.model} "
        "-i {output.info} "
        "--interv-key {params.interv} "
        "{params.use_covariate} "
        "{params.use_size} "
        "{params.use_layer} "
        "{params.tune_ct} "
        "--batch-size {params.bs} "
        "--n-devices {resources.gpu} "
        "--log-subdir tune_ct={wildcards.tune_ct} "
        "--random-sleep 60 "
        "-v "
        "&> {log} || touch {params.blacklist}"


rule cascade_ctfact:
    input:
        data=join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "ctfact_{phs}.h5ad"),
        model=tune_model_switch,
    output:
        data=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_CTFACT_CFG,
            "ctfact_{phs}.h5ad",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_CTFACT_CFG,
            "info_ctfact_{phs}.yaml",
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
            "cascade",
            CASCADE_CTFACT_CFG,
            "cascade_ctfact_{phs}.log",
        ),
    threads: 8
    resources:
        gpu=1,
    params:
        interv=get_interv,
        use_covariate=get_use_covariate,
        use_size=get_use_size,
        use_layer=get_use_layer,
        ablation=get_ablation,
        bs=get_bs(128),
    shell:
        "{config[cascade_prefix]}"
        "{config[cmd_prefix]}"
        "cascade counterfactual "
        "-d {input.data} "
        "-m {input.model} "
        "-p {output.data} "
        "-i {output.info} "
        "--interv-key {params.interv} "
        "{params.use_covariate} "
        "{params.use_size} "
        "{params.use_layer} "
        "{params.ablation} "
        "--batch-size {params.bs} "
        "--n-devices {resources.gpu} "
        "-v "
        "&> {log}"


rule cascade_dsgn_df:
    input:
        ctrl=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=tune_model_switch,
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
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "df"),
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "df"),
            "dsgn_{phs}",
            "{name}",
            "info_dsgn.yaml",
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
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "df"),
            "dsgn_{phs}",
            "{name}",
            "cascade_dsgn_df.log",
        ),
    threads: 8
    resources:
        gpu=1,
    params:
        interv=get_interv,
        use_covariate=get_use_covariate,
        use_size=get_use_size,
        use_layer=get_use_layer,
        size=get_design_size,
        bs=get_bs(32),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "df"),
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[cascade_prefix]}"
        "{config[cmd_prefix]}"
        "cascade design "
        "-d {input.ctrl} "
        "-m {input.model} "
        "-t {input.target} "
        "--pool {input.pool} "
        "-o {output.design} "
        "-i {output.info} "
        "--interv-key {params.interv} "
        "{params.use_covariate} "
        "{params.use_size} "
        "{params.use_layer} "
        "--design-size {params.size} "
        "--batch-size {params.bs} "
        "--n-devices {resources.gpu} "
        "--log-subdir tune_ct={wildcards.tune_ct}/"
        "name={wildcards.name}-dsgn_sb=False "
        "--random-sleep 60 "
        "-v "
        "&> {log} || touch {params.blacklist}"


rule cascade_dsgn_sb:
    input:
        ctrl=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=tune_model_switch,
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
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "sb"),
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        design_mod=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "sb"),
            "dsgn_{phs}",
            "{name}",
            "design.pt",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "sb"),
            "dsgn_{phs}",
            "{name}",
            "info_dsgn.yaml",
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
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "sb"),
            "dsgn_{phs}",
            "{name}",
            "cascade_dsgn_sb.log",
        ),
    threads: 8
    resources:
        gpu=1,
    params:
        interv=get_interv,
        use_covariate=get_use_covariate,
        use_size=get_use_size,
        use_layer=get_use_layer,
        size=get_design_size,
        bs=get_bs(32),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "sb"),
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[cascade_prefix]}"
        "{config[cmd_prefix]}"
        "cascade design "
        "-d {input.ctrl} "
        "-m {input.model} "
        "-t {input.target} "
        "--pool {input.pool} "
        "-o {output.design} "
        "-u {output.design_mod} "
        "-i {output.info} "
        "--interv-key {params.interv} "
        "{params.use_covariate} "
        "{params.use_size} "
        "{params.use_layer} "
        "--design-size {params.size} "
        "--design-scale-bias "
        "--batch-size {params.bs} "
        "--n-devices {resources.gpu} "
        "--log-subdir tune_ct={wildcards.tune_ct}/"
        "name={wildcards.name}-dsgn=sb "
        "--random-sleep 60 "
        "-v "
        "&> {log} || touch {params.blacklist}"


rule cascade_dsgn_bf:
    input:
        ctrl=join("dat", "{ctx}", "{dat}", "{sub}", "ctrl.h5ad"),
        model=tune_model_switch,
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
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "bf"),
            "dsgn_{phs}",
            "{name}",
            "dsgn.csv",
        ),
        info=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "bf"),
            "dsgn_{phs}",
            "{name}",
            "info_dsgn.yaml",
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
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "bf"),
            "dsgn_{phs}",
            "{name}",
            "cascade_dsgn_bf.log",
        ),
    threads: 8
    resources:
        gpu=1,
    params:
        interv=get_interv,
        use_covariate=get_use_covariate,
        use_size=get_use_size,
        use_layer=get_use_layer,
        size=get_design_size,
        bs=get_bs(128),
        blacklist=join(
            "inf",
            "{ctx}",
            "{dat}",
            "{sub}",
            "{scf}",
            "{aux}",
            "{div}",
            "cascade",
            CASCADE_DSGN_CFG.replace("{dsgn}", "bf"),
            "dsgn_{phs}",
            "{name}",
            ".blacklist",
        ),
    shell:
        "{config[cascade_prefix]}"
        "{config[cmd_prefix]}"
        "cascade design_brute_force "
        "-d {input.ctrl} "
        "-m {input.model} "
        "-t {input.target} "
        "--pool {input.pool} "
        "-o {output.design} "
        "-i {output.info} "
        "--interv-key {params.interv} "
        "{params.use_covariate} "
        "{params.use_size} "
        "{params.use_layer} "
        "--design-size {params.size} "
        "--batch-size {params.bs} "
        "--n-devices {resources.gpu} "
        "--random-sleep 60 "
        "-v "
        "&> {log} || touch {params.blacklist}"
