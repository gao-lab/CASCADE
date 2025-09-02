#!/bin/bash
set -e
export LC_ALL=C
zcat JASPAR2022-hg38-merge.bed.gz | sort -k 1,1 -k2,2n | gzip > JASPAR2022-hg38-merge.sorted.bed.gz
