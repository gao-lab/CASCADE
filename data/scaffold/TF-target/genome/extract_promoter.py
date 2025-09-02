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
    gtf = gtf.assign(**attributes)
    gtf["start"] -= 1
    pos_strand = gtf.query("strand == '+'").copy()
    neg_strand = gtf.query("strand == '-'").copy()
    pos_strand["start"], pos_strand["end"] = (
        (pos_strand["start"] - 2000).clip(lower=0),
        pos_strand["start"] + 500,
    )
    neg_strand["start"], neg_strand["end"] = (
        (neg_strand["end"] - 500).clip(lower=0),
        neg_strand["end"] + 2000,
    )
    pos_strand = pos_strand.loc[:, ["seqname", "start", "end", "gene_name"]]
    neg_strand = neg_strand.loc[:, ["seqname", "start", "end", "gene_name"]]
    promoter = pd.concat([pos_strand, neg_strand])
    promoter.to_csv(sys.stdout, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
