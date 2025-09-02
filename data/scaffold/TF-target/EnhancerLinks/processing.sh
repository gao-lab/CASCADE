#!/bin/bash

set -e
export LC_ALL=C
zcat merge_links_allAnno.bed.gz | tail -n+2 | cut -f1-4 | ./id2name.py | sort -k 1,1 -k2,2n | gzip > merge_links_allAnno.brief.sorted.bed.gz
