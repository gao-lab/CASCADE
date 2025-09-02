#!/bin/bash

set -e

gunzip -k GSE217460_210322_TFAtlas_subsample_raw.h5ad.gz
python collect.py
