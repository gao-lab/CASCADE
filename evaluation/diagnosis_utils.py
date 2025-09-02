import subprocess
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import pymde
import seaborn as sns
import torch
from IPython.display import display
from ipywidgets import ToggleButtons
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from pandas.api.types import is_numeric_dtype
from scipy.sparse import csr_matrix, issparse
from tqdm.auto import tqdm

args = {
    "ds": "Replogle-2022-K562-ess",
    "imp": "20",
    "n_vars": "2000",
    "skt": "kegg+tf+ppi+corr",
    "aux": "lsi",
    "kg": "0.9",
    "kc": "0.75",
    "div_sd": "0",
    "nptc": "4",
    "dz": "8",
    "beta": "0.1",
    "sps": "L1",
    "acyc": "SpecNorm",
    "lik": "NegBin",
    "lam": "0.1",
    "alpha": "0.5",
    "run_sd": "0",
}  # Default


opts = {
    "ds": [
        "Adamson-2016",
        "Replogle-2022-K562-ess",
        "Replogle-2022-K562-gwps",
        "Replogle-2022-RPE1",
        "Norman-2019",
    ],
    "imp": ["0", "10", "20"],
    "n_vars": ["200", "2000"],
    "skt": ["cpl", "kegg+tf+ppi+corr"],
    "aux": ["go", "svd", "scgpt", "lsi"],
    "kg": ["0.9", "1.0"],
    "kc": ["0.75", "1.0"],
    "div_sd": [str(i) for i in range(5)],
    "nptc": ["1", "4", "8"],
    "dz": ["0", "4", "8"],
    "beta": ["0.1"],
    "sps": ["ScaleFree", "L1"],
    "acyc": ["TrExp", "SpecNorm", "LogDet"],
    "lik": ["Normal", "NegBin"],
    "lam": ["0.1"],
    "alpha": ["0.5"],
    "run_sd": [str(i) for i in range(4)],
}


def on_click(change):
    global args
    for k, toggle in toggles.items():
        if change.owner is toggle:
            args[k] = change["new"]
            break
    update_args()


def update_args():
    global args
    ds = f"ds={args['ds']}"
    imp = f"imp={args['imp']}"
    n_vars = f"n_vars={args['n_vars']}"
    skt, aux = args["skt"], args["aux"]
    div = f"kg={args['kg']}-kc={args['kc']}-div_sd={args['div_sd']}"
    run = (
        f"nptc={args['nptc']}-dz={args['dz']}-beta={args['beta']}-"
        f"sps={args['sps']}-acyc={args['acyc']}-lik={args['lik']}-"
        f"lam={args['lam']}-alp={args['alpha']}-run_sd={args['run_sd']}"
    )

    wk_dir = Path("dat") / ds / imp / n_vars
    args["scaffold"] = wk_dir / skt / "scaffold.gml.gz"

    dat_dir = wk_dir / div
    args["train"] = dat_dir / "train.h5ad"
    args["test"] = dat_dir / "test.h5ad"
    args["ctfact_train_input"] = dat_dir / "ctfact_train.h5ad"
    args["ctfact_test_input"] = dat_dir / "ctfact_test.h5ad"

    inf_dir = Path("inf") / ds / imp / n_vars / skt / aux / div / "cascade" / run
    args["discover"] = inf_dir / "discover.pt"
    inf_dir = inf_dir.with_name(f"{inf_dir.name}-tune_ct=True")
    args["tune"] = inf_dir / "tune.pt"
    args["ctfact_train"] = inf_dir / "ctfact_train.h5ad"
    args["ctfact_test"] = inf_dir / "ctfact_test.h5ad"
    args["metrics_ctfact_train"] = inf_dir / "metrics_ctfact_train_each.csv"
    args["metrics_ctfact_test"] = inf_dir / "metrics_ctfact_test_each.csv"


update_args()
toggles = {
    k: ToggleButtons(options=opts[k], label=args[k], description=k) for k in opts
}


def show_toggles():
    for toggle in toggles.values():
        toggle.observe(on_click, names="value")
        display(toggle)


category_norm = Normalize(vmin=-0.5, vmax=6.5)
category_cmap = ListedColormap(
    [
        "#adc8e6",  # "0 seen"
        "#1778b1",  # "1 seen"
        "#19bfce",  # "2 seen"
        "#ff7e28",  # "0/2 unseen"
        "#d6262c",  # "1/2 unseen"
        "#e278c0",  # "2/2 unseen"
        "#a647f7",  # "1/1 unseen"
    ],
    name="category",
)
category_cmap.set_bad("lightgrey")
discrete_cmap = sns.color_palette("tab20", as_cmap=True)
discrete_cmap.set_bad("lightgrey")
numeric_cmap = sns.color_palette("viridis", as_cmap=True)
numeric_cmap.set_bad("lightgrey")


def densify(x):
    return x.toarray() if issparse(x) else x


def split_mask(df, hue, split):
    if split == "train":
        df["mask"] = df["category"].str.endswith(" seen")
    elif split == "test":
        df["mask"] = df["category"].str.endswith(" unseen")
    elif split == "ctrl":
        df["mask"] = df["category"] == "0 seen"
    else:  # split in {"both", "all"}
        df["mask"] = True

    df.loc[~df["mask"], hue] = np.nan
    df.sort_values(hue, na_position="first", inplace=True)
    del df["mask"]


def scatter(x, y, c, data=None, log=False, ax=None, **kwargs):
    if data is not None:
        xlab, ylab, clab, x, y, c = x, y, c, data[x], data[y], data[c]
    else:
        xlab, ylab, clab = x.name, y.name, c.name
    if is_numeric_dtype(c):
        cmap = numeric_cmap
        norm = LogNorm() if log else Normalize()
    else:
        cmap = category_cmap if clab == "category" else discrete_cmap
        norm = category_norm if clab == "category" else Normalize()
        c, c_ = c.cat.codes.replace(-1, np.nan), c

    ax = ax or plt.gca()
    fig = ax.get_figure()
    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, plotnonfinite=True, **kwargs)
    ax.set_xlabel(xlab or "")
    ax.set_ylabel(ylab or "")
    cax = fig.add_axes([1.0, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(sc, cax=cax, label=clab or "")
    if cmap is not numeric_cmap:
        cbar.set_ticks(list(range(c_.cat.categories.size)))
        cbar.set_ticklabels(c_.cat.categories)


def mde(df, emb_dim=2, clip_pct=0.9, abs_norm=2, dissimilar_ratio=1, seed=42):
    assert (df.index == df.columns).all()
    x = df.to_numpy()
    upper = np.quantile(x[x > 0], clip_pct)
    lower = np.quantile(x[x < 0], 1 - clip_pct)
    x = x.clip(min=lower, max=upper)
    print(f"Clipping at lower = {lower:.5f}, upper = {upper:.5f}")

    x = csr_matrix(x)
    x = x + x.T  # Make symmetric
    s = abs_norm / max(np.fabs(x.min()), np.fabs(x.max()))
    x = x * s
    print(f"Abs normalized to min = {x.min():.5f}, max = {x.max():.5f}")

    pymde.seed(seed)
    graph = pymde.Graph(x)
    dissimilar_edges = pymde.preprocess.dissimilar_edges(
        graph.n_items, graph.edges, num_edges=round(graph.n_edges * dissimilar_ratio)
    )
    combined_graph = pymde.Graph.from_edges(
        torch.cat([graph.edges, dissimilar_edges]),
        weights=torch.cat(
            [
                graph.weights,
                -1 * graph.weights.new_ones(dissimilar_edges.size(0)),
            ]
        ),
    )

    f = pymde.penalties.PushAndPull(
        weights=combined_graph.weights,
        attractive_penalty=pymde.penalties.Log1p,
        repulsive_penalty=pymde.penalties.Log,
    )
    mde = pymde.MDE(
        n_items=df.shape[0],
        embedding_dim=emb_dim,
        edges=combined_graph.edges,
        distortion_function=f,
        constraint=pymde.Standardized(),
    )
    emb = pd.DataFrame(
        mde.embed(snapshot_every=3).numpy(),
        index=df.index,
        columns=[f"MDE{i+1}" for i in range(emb_dim)],
    )
    mde.play(figsize_inches=(3, 3))
    mde.distortions_cdf()
    return emb


def kde_scatter(x, y, c, data, yerr=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        sns.kdeplot(
            data=data.sample(n=min(10000, data.shape[0]), replace=False),
            x=x,
            y=y,
            hue=c,
            common_norm=False,
            warn_singular=False,
            fill=False,
            ax=ax,
        )
    except ValueError:
        pass
    groups = [(i, f) for i, f in data.groupby(c, observed=True, dropna=False)]
    nan_groups = [(i, f) for i, f in groups if i is np.nan]
    non_nan_groups = [(i, f) for i, f in groups if i is not np.nan]
    for i, f in [*nan_groups, *non_nan_groups]:
        kws = {"c": "lightgrey"} if i is np.nan else {}
        ax.scatter(x=f[x], y=f[y], s=2, alpha=0.2, rasterized=True, **kws)
        if yerr is None:
            continue
        ax.errorbar(
            x=f[x],
            y=f[y],
            yerr=f[yerr],
            capsize=0,
            linestyle="None",
            elinewidth=1,
            alpha=0.05,
            rasterized=True,
            **kws,
        )
    return fig, ax


uniprot_map = pd.read_table(
    "../data/function/GO/goa_human.gaf",
    comment="!",
    usecols=[1, 2],
    names=["uniprot_id", "symbol"],
).drop_duplicates()
symbol2uniprot = uniprot_map.set_index("symbol")["uniprot_id"].to_dict()
uniprot2symbol = uniprot_map.set_index("uniprot_id")["symbol"].to_dict()
goea_dir = Path("goea")


def run_goea(leiden):
    if goea_dir.exists():
        rmtree("goea")
    goea_dir.mkdir(parents=True)
    bg = leiden.index
    np.savetxt(goea_dir / "bg.txt", bg, fmt="%s")
    np.savetxt(goea_dir / "bg_id.txt", bg.map(symbol2uniprot).dropna(), fmt="%s")
    for i in leiden.cat.categories:
        fg = leiden.index[leiden == i]
        np.savetxt(goea_dir / f"fg{i}.txt", fg, fmt="%s")
        np.savetxt(
            goea_dir / f"fg{i}_id.txt", fg.map(symbol2uniprot).dropna(), fmt="%s"
        )

    for i in tqdm(leiden.cat.categories):
        with (goea_dir / f"goea{i}.log").open("w") as f:
            subprocess.run(
                [
                    "find_enrichment.py",
                    goea_dir / f"fg{i}_id.txt",
                    goea_dir / "bg_id.txt",
                    "../data/function/GO/goa_human.gaf",
                    "--obo",
                    "../data/function/GO/go-basic.obo",
                    "--goslim",
                    "../data/function/GO/goslim_generic.obo",
                    "--method",
                    "fdr_bh",
                    "--outfile",
                    goea_dir / f"goea{i}.xlsx",
                ],
                stdout=f,
                stderr=f,
            )


def get_goea_df(i):
    df = (
        pd.read_excel(goea_dir / f"goea{i}.xlsx")
        .query("enrichment == 'e' & depth > 2 & p_fdr_bh < 0.05")
        .sort_values("p_fdr_bh")
    )
    df["study_items"] = df["study_items"].map(
        lambda x: ", ".join(uniprot2symbol.get(i, i) for i in x.split(", "))
    )
    df["-log10_fdr"] = -np.log10(df["p_fdr_bh"])
    return df
