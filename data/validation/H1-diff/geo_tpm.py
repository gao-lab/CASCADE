#!/usr/bin/env python

import pandas as pd

genes = pd.read_table(
    "bulkRNA/resources/human/GENCODE_v37/genes/"
    "gencodeV37_geneid_60710_uniquename_59453.txt"
)

tpm = (
    pd.read_table(
        "bulkRNA/results/03_quant/s02_tpm.txt",
        usecols=[
            "gene_id",
            "gene_name",
            "gene_unique_name",
            "20250527-GTZ-1-2",
            "20250527-GTZ-1-3",
            "20250602-GTZ-1-1",
            "20250602-GTZ-1-2",
            "20250527-GTZ-1-4",
            "20250527-GTZ-2-2",
            "20250529-GTZ-1-2",
            "20250527-GTZ-1-1",
            "20250706-GTZ-1-1",
            "20250706-GTZ-1-2",
            "20250716-GTZ-1-1",
            "20250716-GTZ-1-2",
            "20250716-GTZ-1-3",
            "20250716-GTZ-1-4",
            "20250527-GTZ-2-1",
            "20250701-GTZ-1-1",
            "20250701-GTZ-1-2",
        ],
    )
    .merge(genes)
    .rename(
        columns={
            "20250527-GTZ-1-2": "hESC-rep1",
            "20250527-GTZ-1-3": "hESC-rep2",
            "20250602-GTZ-1-1": "iNPC-d4-rep1",
            "20250602-GTZ-1-2": "iNPC-d4-rep2",
            "20250527-GTZ-1-4": "iNPC-d7-rep1",
            "20250527-GTZ-2-2": "iNPC-d7-rep2",
            "20250529-GTZ-1-2": "iNPC-d7-rep3",
            "20250527-GTZ-1-1": "PAX6-rep1",
            "20250706-GTZ-1-1": "PAX6-rep2",
            "20250706-GTZ-1-2": "PAX6-rep3",
            "20250716-GTZ-1-1": "TFAP2A-rep1",
            "20250716-GTZ-1-2": "TFAP2A-rep2",
            "20250716-GTZ-1-3": "TFAP2A-rep3",
            "20250716-GTZ-1-4": "TFAP2A-rep4",
            "20250527-GTZ-2-1": "PAX6+TFAP2A-rep1",
            "20250701-GTZ-1-1": "PAX6+TFAP2A-rep2",
            "20250701-GTZ-1-2": "PAX6+TFAP2A-rep3",
        }
    )
    .loc[
        :,
        [
            "gene_id",
            "gene_unique_name",
            "gene_type",
            "hESC-rep1",
            "hESC-rep2",
            "iNPC-d4-rep1",
            "iNPC-d4-rep2",
            "iNPC-d7-rep1",
            "iNPC-d7-rep2",
            "iNPC-d7-rep3",
            "PAX6-rep1",
            "PAX6-rep2",
            "PAX6-rep3",
            "TFAP2A-rep1",
            "TFAP2A-rep2",
            "TFAP2A-rep3",
            "TFAP2A-rep4",
            "PAX6+TFAP2A-rep1",
            "PAX6+TFAP2A-rep2",
            "PAX6+TFAP2A-rep3",
        ],
    ]
)
tpm.to_csv("GEO/tpm.csv.gz", index=False)
