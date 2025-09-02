#!/usr/bin/env python

import sys

import pandas as pd


def main():
    id2name = pd.read_table(
        "../genome/id2name.txt", sep="\t", names=["gene_id", "gene_name"]
    )
    id2name = {
        gene_id: gene_name
        for gene_id, gene_name in zip(id2name["gene_id"], id2name["gene_name"])
    }
    for line in sys.stdin:
        split = line.strip().split("\t")
        split[-1] = id2name.get(split[-1], split[-1])
        print("\t".join(split))


if __name__ == "__main__":
    main()
