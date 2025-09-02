#!/bin/bash
set -e
export LC_ALL=C
bedtools intersect -a ../genome/gencode.v44.promoters.sorted.bed.gz -b ../MotifScan/JASPAR2022-hg38-merge.sorted.bed.gz -wo -sorted | gzip > promoter-tf.bed.gz
bedtools intersect -a ../EnhancerLinks/merge_links_allAnno.brief.sorted.bed.gz -b ../MotifScan/JASPAR2022-hg38-merge.sorted.bed.gz -wo -sorted | gzip > enhancer-tf.bed.gz
zcat promoter-tf.bed.gz | cut -f4,8 | sort | uniq | gzip > promoter-tf.txt.gz
zcat enhancer-tf.bed.gz | cut -f4,8 | sort | uniq | gzip > enhancer-tf.txt.gz
zcat promoter-tf.txt.gz enhancer-tf.txt.gz | sort | uniq | gzip > gene-tf.txt.gz
