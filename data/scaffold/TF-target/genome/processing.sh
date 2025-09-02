#!/bin/bash
set -e
export LC_ALL=C
zcat gencode.v44.chr_patch_hapl_scaff.basic.annotation.gtf.gz | ./extract_promoter.py | sort -k 1,1 -k2,2n | gzip > gencode.v44.promoters.sorted.bed.gz
zcat gencode.v44.chr_patch_hapl_scaff.basic.annotation.gtf.gz | ./extract_id2name.py > id2name.txt
