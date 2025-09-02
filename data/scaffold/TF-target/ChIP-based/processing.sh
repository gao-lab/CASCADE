#!/bin/bash
set -e
export LC_ALL=C
bedtools intersect \
    -a ../genome/gencode.v44.promoters.sorted.bed.gz \
    -b ../ENCODE-ChIP/ENCODE-TF-ChIP-GRCh38.bed.gz \
    -wo -f 0.75 -F 0.75 -e -sorted | gzip > promoter-tf.bed.gz &
bedtools intersect \
    -a ../EnhancerLinks/merge_links_allAnno.brief.sorted.bed.gz \
    -b ../ENCODE-ChIP/ENCODE-TF-ChIP-GRCh38.bed.gz \
    -wo -f 0.75 -F 0.75 -e -sorted | gzip > enhancer-tf.bed.gz &
wait

zcat promoter-tf.bed.gz | cut -f4,8 | sort | uniq -c | awk 'BEGIN{OFS=","} {print($3, $2, $1)}' | gzip > tf-promoter.csv.gz &
zcat enhancer-tf.bed.gz | cut -f4,8 | sort | uniq -c | awk 'BEGIN{OFS=","} {print($3, $2, $1)}' | gzip > tf-enhancer.csv.gz &
zcat promoter-tf.bed.gz enhancer-tf.bed.gz | cut -f4,8 | sort | uniq -c | awk 'BEGIN{OFS=","} {print($3, $2, $1)}' | gzip > tf-gene.csv.gz &
wait
