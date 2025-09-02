# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from glob import glob
from os.path import basename, dirname

import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

from cascade.metrics import dsgn_hrc_exact
from cascade.plot import set_figure_params

# %%
set_figure_params()

# %% [markdown]
# # Read data

# %%
with open("config/display.yaml") as f:
    display = yaml.load(f, Loader=yaml.Loader)
meth_lut = {v: k for k, v in display["naming"]["methods"].items()}

# %%
additive = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["votes"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/cpl/nil/kg=0.9-kc=0.75-div_sd=*/"
        "additive/run/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(additive)
additive = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["additive"]})

# %%
linear = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["votes"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/cpl/nil/kg=0.9-kc=0.75-div_sd=*/"
        "linear/dim=10-lam=0.1-run_sd=0/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(linear)
linear = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["linear"]})

# %%
cpa = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["votes"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/cpl/lsi/kg=0.9-kc=0.75-div_sd=*/"
        "cpa/run_sd=0/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(cpa)
cpa = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["cpa"]})

# %%
biolord = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["votes"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/cpl/lsi/kg=0.9-kc=0.75-div_sd=*/"
        "biolord/run_sd=0/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(biolord)
biolord = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["biolord"]})

# %%
gears = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["votes"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/cpl/go/kg=0.9-kc=0.75-div_sd=*/"
        "gears/hidden_size=64-epochs=20-run_sd=0/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(gears)
gears = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["gears"]})

# %%
scgpt = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["votes"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/cpl/go/kg=0.9-kc=0.75-div_sd=*/"
        "scgpt/epochs=20-run_sd=0/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(scgpt)
scgpt = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["scgpt"]})

# %%
scfoundation = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["votes"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/cpl/go/kg=0.9-kc=0.75-div_sd=*/"
        "scfoundation/hidden_size=512-epochs=20-run_sd=0/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(scfoundation)
scfoundation = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["scfoundation"]})

# %%
cascade_sb = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["score"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/kegg+tf+ppi+corr/lsi/kg=0.9-kc=0.75-div_sd=*/"
        "cascade/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        "lam=0.1-alp=0.5-run_sd=0-tune_ct=True-dsgn=sb/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(cascade_sb)
cascade_sb = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["cascade_sb"]})

# %%
cascade_bf = {
    basename(dirname(file)).replace("+", ","): pd.read_csv(
        file, index_col=0, keep_default_na=False
    )["votes"]
    for file in glob(
        "inf/ds=*/imp=20/n_vars=2000/kegg+tf+ppi+corr/lsi/kg=0.9-kc=0.75-div_sd=*/"
        "cascade/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        "lam=0.1-alp=0.5-run_sd=0-tune_ct=True-dsgn=bf/dsgn_test/*/dsgn.csv"
    )
}
qtl, hr = dsgn_hrc_exact(cascade_bf)
cascade_bf = pd.DataFrame({"qtl": qtl, "hr": hr, "Method": meth_lut["cascade_bf"]})

# %% [markdown]
# # Plot

# %%
df = pd.concat(
    [additive, linear, cpa, biolord, gears, scgpt, scfoundation, cascade_bf, cascade_sb]
)

# %%
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.axline((0, 0), slope=1, color="lightgrey", linestyle="--")
sns.lineplot(
    data=df,
    x="qtl",
    y="hr",
    hue="Method",
    estimator=None,
    palette=display["palette"]["methods"],
    legend=False,
    ax=ax,
)
ax.set_xlabel("Design quantile")
ax.set_ylabel("Hit rate")
fig.savefig("dsgn_cmp.pdf")
