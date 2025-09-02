from os.path import join


# ---------------------------- Simulation specific -----------------------------


rule gen_dag:
    output:
        join(
            "dat",
            "n_vars={n_vars}-in_deg={in_deg}-gph_tp={gph_tp}-gph_sd={gph_sd}",
            "causal.gml.gz",
        ),
    log:
        join(
            "dat",
            "n_vars={n_vars}-in_deg={in_deg}-gph_tp={gph_tp}-gph_sd={gph_sd}",
            "gen_dag.log",
        ),
    shell:
        "python workflow/scripts/gen_dag.py "
        "--n-vars {wildcards.n_vars} "
        "--in-degree {wildcards.in_deg} "
        "--type {wildcards.gph_tp} "
        "--seed {wildcards.gph_sd} "
        "--output {output} "
        "&> {log}"


rule sim_data:
    input:
        join("{path}", "causal.gml.gz"),
    output:
        join(
            "{path}",
            "n_obs={n_obs}-int_f={int_f}-act={act}-snr={snr}-dat_sd={dat_sd}",
            "data.h5ad",
        ),
    log:
        join(
            "{path}",
            "n_obs={n_obs}-int_f={int_f}-act={act}-snr={snr}-dat_sd={dat_sd}",
            "sim_data.log",
        ),
    shell:
        "python workflow/scripts/sim_data.py "
        "--input {input} "
        "--output {output} "
        "--n-obs {wildcards.n_obs} "
        "--int-frac {wildcards.int_f} "
        "--act {wildcards.act} "
        "--snr {wildcards.snr} "
        "--seed {wildcards.dat_sd} "
        "&> {log}"


rule subset_vars:
    input:
        graph=join("{path}", "causal.gml.gz"),
        data=join("{path}", "{dat}", "data.h5ad"),
    output:
        graph=join("{path}", "{dat}", "sub_f={sub_f}-sub_sd={sub_sd}", "sub.gml.gz"),
        data=join("{path}", "{dat}", "sub_f={sub_f}-sub_sd={sub_sd}", "sub.h5ad"),
    log:
        join(
            "{path}",
            "{dat}",
            "sub_f={sub_f}-sub_sd={sub_sd}",
            "subset_vars.log",
        ),
    shell:
        "python workflow/scripts/subset_vars.py "
        "--input-graph {input.graph} "
        "--input-data {input.data} "
        "--output-graph {output.graph} "
        "--output-data {output.data} "
        "--frac {wildcards.sub_f} "
        "--seed {wildcards.sub_sd} "
        "&> {log}"


rule build_sim_scaffold_graph:
    input:
        join("{path}", "sub.gml.gz"),
    output:
        join(
            "{path}",
            "tpr={tpr}-fpr={fpr}-scf_sd={scf_sd}",
            "scaffold.gml.gz",
        ),
    log:
        join(
            "{path}",
            "tpr={tpr}-fpr={fpr}-scf_sd={scf_sd}",
            "build_sim_scaffold_graph.log",
        ),
    shell:
        "python workflow/scripts/build_sim_scaffold_graph.py "
        "--input {input} "
        "--output {output} "
        "--tpr {wildcards.tpr} "
        "--fpr {wildcards.fpr} "
        "--seed {wildcards.scf_sd} "
        "&> {log}"


# --------------------------- Actual data specific -----------------------------


rule link_causal:
    input:
        lambda wildcards: config["datasets"][wildcards.ds]["gph"],
    output:
        join("dat", "ds={ds}", "causal.gml.gz"),
    localrule: True
    shell:
        "ln -fnrs {input} {output}"


rule impute_data:
    input:
        data=lambda wildcards: config["datasets"][wildcards.ds]["dat"],
    output:
        join("dat", "ds={ds}", "imp={imp}", "data.h5ad"),
    log:
        join("dat", "ds={ds}", "imp={imp}", "impute_data.log"),
    shell:
        "python workflow/scripts/impute_data.py "
        "--input {input} "
        "--output {output} "
        "-k {wildcards.imp} "
        "&> {log}"


rule select_vars:
    input:
        graph=join("{path}", "causal.gml.gz"),
        data=join("{path}", "{dat}", "data.h5ad"),
    output:
        graph=join("{path}", "{dat}", "n_vars={n_vars}", "sub.gml.gz"),
        data=join("{path}", "{dat}", "n_vars={n_vars}", "sub.h5ad"),
    log:
        join("{path}", "{dat}", "n_vars={n_vars}", "select_vars.log"),
    shell:
        "python workflow/scripts/select_vars.py "
        "--input-graph {input.graph} "
        "--input-data {input.data} "
        "--output-graph {output.graph} "
        "--output-data {output.data} "
        "--n-vars {wildcards.n_vars} "
        "&> {log}"


rule build_actual_scaffold_graph:
    input:
        data=join("dat", "ds={ds}", "{dat}", "{sub}", "sub.h5ad"),
        graph=lambda wildcards: [
            config["scf_resources"][g] for g in wildcards.scf.split("+")
        ],
    output:
        join("dat", "ds={ds}", "{dat}", "{sub}", "{scf}", "scaffold.gml.gz"),
    log:
        join(
            "dat",
            "ds={ds}",
            "{dat}",
            "{sub}",
            "{scf}",
            "build_actual_scaffold_graph.log",
        ),
    wildcard_constraints:
        scf="(kegg|tf|ppi|corr).*",
    shell:
        "python workflow/scripts/build_actual_scaffold_graph.py "
        "--input-data {input.data} "
        "--input-graph {input.graph} "
        "--output {output} "
        "&> {log}"


rule build_svd_latent_data:
    input:
        join("dat", "ds={ds}", "{dat}", "{sub}", "sub.h5ad"),
    output:
        join("dat", "ds={ds}", "{dat}", "{sub}", "svd", "latent.csv.gz"),
    log:
        join("dat", "ds={ds}", "{dat}", "{sub}", "svd", "build_svd_latent_data.log"),
    shell:
        "python workflow/scripts/build_svd_latent_data.py "
        "--input {input} "
        "--output {output} "
        "&> {log}"


rule build_emb_latent_data:
    input:
        data=join("dat", "ds={ds}", "{dat}", "{sub}", "sub.h5ad"),
        emb=lambda wildcards: config["aux_resources"][wildcards.aux],
    output:
        join("dat", "ds={ds}", "{dat}", "{sub}", "{aux}", "latent.csv.gz"),
    log:
        join("dat", "ds={ds}", "{dat}", "{sub}", "{aux}", "build_emb_latent_data.log"),
    wildcard_constraints:
        aux="lsi|scgpt",
    shell:
        "python workflow/scripts/build_emb_latent_data.py "
        "--input-data {input.data} "
        "--input-emb {input.emb} "
        "--output {output} "
        "&> {log}"


rule build_go_latent_data:
    input:
        data=join("dat", "ds={ds}", "{dat}", "{sub}", "sub.h5ad"),
        graph=lambda wildcards: config["aux_resources"]["go"],
    output:
        join(
            "dat",
            "ds={ds}",
            "{dat}",
            "{sub}",
            "go",
            "latent.gml.gz",
        ),
    log:
        join(
            "dat",
            "ds={ds}",
            "{dat}",
            "{sub}",
            "go",
            "build_go_latent_data.log",
        ),
    shell:
        "python workflow/scripts/build_go_latent_data.py "
        "--input-data {input.data} "
        "--input-graph {input.graph} "
        "--output {output} "
        "&> {log}"


rule cp_go_files:  # Avoid singularity breaking symlinks
    input:
        config["aux_resources"]["go_path"],
    output:
        directory(join("dat", "go_files")),
    localrule: True
    shell:
        "mkdir {output} && "
        "cp {input}/gene2go_all.pkl {output}/gene2go_all.pkl && "
        "cp {input}/essential_all_data_pert_genes.pkl {output}/essential_all_data_pert_genes.pkl && "
        "cp -r {input}/go_essential_all {output}/go_essential_all"


rule cp_lsi:  # Avoid singularity breaking symlinks
    input:
        config["aux_resources"]["lsi"],
    output:
        join("dat", "gene2gos_lsi.csv.gz"),
    localrule: True
    shell:
        "cp {input} {output}"


checkpoint request_designs:
    input:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "{phs}.h5ad"),
    output:
        directory(
            join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "request_designs_{phs}")
        ),
    log:
        join("dat", "{ctx}", "{dat}", "{sub}", "{div}", "request_designs_{phs}.log"),
    shell:
        "python workflow/scripts/request_designs.py "
        "--input {input} "
        "--output {output} "
        "-k {config[top_design]} "
        "&> {log}"


# ---------------------------------- Common ------------------------------------


rule build_complete_scaffold_graph:
    input:
        join("{path}", "sub.h5ad"),
    output:
        join("{path}", "cpl", "scaffold.gml.gz"),
    log:
        join("{path}", "cpl", "build_complete_scaffold_graph.log"),
    shell:
        "python workflow/scripts/build_complete_scaffold_graph.py "
        "--input {input} "
        "--output {output} "
        "&> {log}"


rule extract_ctrl:
    input:
        join("{path}", "sub.h5ad"),
    output:
        join("{path}", "ctrl.h5ad"),
    log:
        join("{path}", "extract_ctrl.log"),
    shell:
        "python workflow/scripts/extract_ctrl.py "
        "--input {input} "
        "--output {output} "
        "&> {log}"


rule div_data:
    input:
        join("{path}", "sub.h5ad"),
    output:
        train=join("{path}", "kg={kg}-kc={kc}-div_sd={div_sd}", "train.h5ad"),
        test=join("{path}", "kg={kg}-kc={kc}-div_sd={div_sd}", "test.h5ad"),
    log:
        join("{path}", "kg={kg}-kc={kc}-div_sd={div_sd}", "div_data.log"),
    shell:
        "python workflow/scripts/div_data.py "
        "--input {input} "
        "--output-train {output.train} "
        "--output-test {output.test} "
        "--kg {wildcards.kg} "
        "--kc {wildcards.kc} "
        "--seed {wildcards.div_sd} "
        "&> {log}"


rule prep_ctfact:
    input:
        ctrl=join("dat", "{path}", "ctrl.h5ad"),
        data=join("dat", "{path}", "{div}", "{phs}.h5ad"),
    output:
        join("dat", "{path}", "{div}", "ctfact_{phs}.h5ad"),
    log:
        join("dat", "{path}", "{div}", "prep_ctfact_{phs}.log"),
    shell:
        "python workflow/scripts/prep_ctfact.py "
        "--input-ctrl {input.ctrl} "
        "--input-data {input.data} "
        "--output {output} "
        "&> {log}"


rule prep_dsgn:
    input:
        join("{path}", "{phs}.h5ad"),
    output:
        join("{path}", "dsgn_{phs}", "{name}.h5ad"),
    log:
        join("{path}", "dsgn_{phs}", "prep_dsgn_{name}.log"),
    shell:
        "python workflow/scripts/prep_dsgn.py "
        "--input {input} "
        "--output {output} "
        "&> {log}"


rule prep_dsgn_pool:
    input:
        join("{path}", "{phs}.h5ad"),
    output:
        join("{path}", "dsgn_{phs}", "pool.txt"),
    log:
        join("{path}", "dsgn_{phs}", "prep_dsgn_pool.log"),
    shell:
        "python workflow/scripts/prep_dsgn_pool.py "
        "--input {input} "
        "--output {output} "
        "&> {log}"


rule downgrade_anndata:
    input:
        join("{path}", "{file}.h5ad"),
    output:
        join("{path}", "{file}_downgrade.h5ad"),
    log:
        join("{path}", "downgrade_anndata_{file}.log"),
    singularity:
        "workflow/envs/anndata.sif"
    shell:
        "downgrade-anndata "
        "{input} {output} "
        "&> {log}"
