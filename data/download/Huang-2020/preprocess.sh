#!/bin/bash

set -e

tar xf GSE150774_RAW.tar
tar xf ./*.tar.gz
python collect.py
