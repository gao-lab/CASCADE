#!/usr/bin/env python

import pickle
import re
from csv import QUOTE_NONE

import pandas as pd
from tqdm.auto import tqdm

hgnc = pd.read_table("hgnc.tsv", index_col=0, quoting=QUOTE_NONE)
hgnc_map = {}

for _, row in tqdm(hgnc.iterrows(), total=hgnc.shape[0]):
    row = row.dropna()
    if "Approved symbol" not in row:
        continue
    approved_symbol = row["Approved symbol"]
    approved_name = (
        {row["Approved name"].strip('"')} if "Approved name" in row else set()
    )
    previous_symbols = {
        item for item in re.split(r",\s?", row.get("Previous symbols", "")) if item
    }
    alias_symbols = {
        item for item in re.split(r",\s?", row.get("Alias symbols", "")) if item
    }
    alias_names = {
        item.strip('"')
        for item in re.split(r',\s?(?=")', row.get("Alias names", ""))
        if item
    }
    previous_name = {
        item.strip('"')
        for item in re.split(r',\s?(?=")', row.get("Previous name", ""))
        if item
    }
    for item in (
        approved_name | previous_symbols | alias_symbols | alias_names | previous_name
    ):
        hgnc_map[item.upper()] = approved_symbol
    hgnc_map[approved_symbol] = approved_symbol


with open("hgnc.pkl", "wb") as f:
    pickle.dump(hgnc_map, f)
