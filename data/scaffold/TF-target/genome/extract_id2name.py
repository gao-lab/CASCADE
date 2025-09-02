#!/usr/bin/env python

import re
import sys

import numpy as np
import pandas as pd


def main():
    gtf = (
        pd.read_table(
            sys.stdin,
            sep="\t",
            comment="#",
            names=[
                "seqname",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attribute",
            ],
        )
        .query("feature == 'gene'")
        .copy()
    )
    pattern = re.compile(r'([^\s]+) "([^"]+)";')
    attributes = pd.DataFrame.from_records(
        np.vectorize(lambda x: {key: val for key, val in pattern.findall(x)})(
            gtf["attribute"]
        ),
        index=gtf.index,
    )
    id2name = attributes.loc[:, ["gene_id", "gene_name"]]
    id2name["gene_id"] = id2name["gene_id"].str.replace(r"\.[0-9_-]+$", "", regex=True)
    id2name = id2name.drop_duplicates()
    id2name.to_csv(sys.stdout, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
