#!/bin/bash

set -e

gunzip -k GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
gunzip -k GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
python collect.py
