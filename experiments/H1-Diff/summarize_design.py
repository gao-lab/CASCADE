from argparse import ArgumentParser, Namespace

import pandas as pd
import parse
from snakemake.script import Snakemake


def auto_numeric(x):
    try:
        return pd.to_numeric(x)
    except ValueError:
        return x


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+")
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    args.params = Namespace(pattern=args.pattern)
    args.output = [args.output]
    del args.pattern
    return args


def main(snakemake: Namespace | Snakemake) -> None:
    design = []
    explained = []
    for item in set(snakemake.input):
        if item.endswith("known_drivers.csv"):
            known_drivers = pd.read_csv(item)
            driver_map = {
                k: {
                    g: r if isinstance(r, str) else "<TBD>"
                    for g, r in zip(v["gene"], v["references"])
                }
                for k, v in known_drivers.groupby("cell_type")
            }
            continue
        entry = parse.parse(snakemake.params.pattern, item)
        if entry:
            conf = entry.named
            del conf["file"]
        else:
            continue
        if item.endswith("design.csv"):
            design.append(
                pd.read_csv(item, index_col=0, keep_default_na=False)
                .reset_index()
                .head(n=10)
                .assign(**conf)
            )
        if item.endswith("explained.csv"):
            explained.append(
                pd.read_csv(item, index_col=0, keep_default_na=False)
                .reset_index()
                .assign(**conf)
            )
    design = pd.concat(design, ignore_index=True)
    explained = pd.concat(explained, ignore_index=True)
    df = pd.merge(design, explained, how="outer")
    df = (
        df.rename(columns={"index": "intervention"})
        .assign(references="")
        .apply(auto_numeric)
    )
    last_columns = [
        "intervention",
        "score",
        "individual_exp",
        "additive_exp",
        "combo_exp",
        "synergy_exp",
        "references",
    ]
    conf_columns = [col for col in df.columns if col not in last_columns]
    df = df[conf_columns + last_columns].sort_values(
        [*conf_columns, "score"],
        ascending=[True] * len(conf_columns) + [False],
        ignore_index=True,
    )

    # Format excel file
    size_idx = df.columns.get_loc("size")
    interv_idx = df.columns.get_loc("intervention")
    score_idx = df.columns.get_loc("score")
    additive_idx = df.columns.get_loc("additive_exp")
    combo_idx = df.columns.get_loc("combo_exp")
    synergy_idx = df.columns.get_loc("synergy_exp")
    ref_idx = df.columns.get_loc("references")
    with pd.ExcelWriter(snakemake.output[0], engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Summary")
        hd = writer.book.add_format({"bold": True, "align": "left"})
        hl = writer.book.add_format({"bold": True, "font_color": "red"})
        url = writer.book.add_format({"underline": 1, "color": "blue"})
        num = writer.book.add_format({"num_format": 2})
        pct = writer.book.add_format({"num_format": 10})
        worksheet = writer.sheets["Summary"]

        for idx, col in enumerate(df.columns):
            worksheet.write(0, idx, col, hd)
        for idx, row in df.iterrows():
            genes = row["intervention"].split(",")
            known = driver_map.get(row["target"], {})
            formatted_genes, formatted_refs = [], []
            for i, gene in enumerate(genes):
                if gene in known:
                    formatted_genes.extend([hl, gene])
                    formatted_refs.extend([url, known[gene]])
                else:
                    formatted_genes.append(gene)
                    formatted_refs.append("NA")
                if i < len(genes) - 1:
                    formatted_genes.append(", ")
                    formatted_refs.append(", ")

            if len(formatted_genes) <= 2:  # Either [gene] or [hl, gene]
                worksheet.write(idx + 1, interv_idx, *formatted_genes[::-1])
            else:
                worksheet.write_rich_string(idx + 1, interv_idx, *formatted_genes)
            if len(formatted_refs) <= 2:  # Either [ref] or [url, ref]
                worksheet.write(idx + 1, ref_idx, *formatted_refs[::-1])
            else:
                worksheet.write_rich_string(idx + 1, ref_idx, *formatted_refs)

        scale2 = {
            "type": "2_color_scale",
            "min_color": "#FFFFFF",
            "max_color": "#63BE7B",
        }
        scale3 = {
            "type": "3_color_scale",
            "min_type": "min",
            "mid_type": "num",
            "max_type": "max",
            "mid_value": 0,
            "min_color": "#F8696B",
            "mid_color": "#FFFFFF",
            "max_color": "#63BE7B",
        }
        worksheet.set_column(score_idx, score_idx, None, num)
        worksheet.set_column(additive_idx, synergy_idx, None, pct)
        worksheet.conditional_format(1, size_idx, df.shape[0], size_idx, scale2)
        worksheet.conditional_format(1, score_idx, df.shape[0], score_idx, scale2)
        worksheet.conditional_format(1, additive_idx, df.shape[0], additive_idx, scale3)
        worksheet.conditional_format(1, combo_idx, df.shape[0], combo_idx, scale3)
        worksheet.conditional_format(1, synergy_idx, df.shape[0], synergy_idx, scale3)
        worksheet.autofilter(0, 0, df.shape[0], df.shape[1] - 1)
        worksheet.autofit()
        worksheet.freeze_panes(1, 0)
        worksheet.set_zoom(120)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main(snakemake)
