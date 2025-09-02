#!/usr/bin/env python

import pandas as pd

degs = pd.read_excel("aba7721_tabless1-s16.xlsx", sheet_name="Table_S4", skiprows=1)
degs = degs.query("gene_type == 'protein_coding'")

with open("cell_type_markers.gmt", "w") as f:
    for ct, df in degs.groupby("max.cluster"):
        df = df.sort_values("fold.change", ascending=False).head(500)
        markers = df["gene_short_name"].str.strip("'")
        f.write("\t".join([ct, ct, *markers]) + "\n")
