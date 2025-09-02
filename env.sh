#!/bin/bash

set -e

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    MGR="conda"
else
    echo "Please install conda/mamba first!"
    exit 1
fi
if command -v mamba &> /dev/null; then
    # shellcheck disable=SC1091
    source "${CONDA_EXE%/*}/../etc/profile.d/mamba.sh"
    MGR="mamba"
fi

if [ "${1}" = "create" ]; then
    if [ ! -d ".renv" ]; then mkdir ".renv"; fi
    "${MGR}" env create -p ./conda -f conda.yaml
    (
        "${MGR}" activate ./conda
        "${MGR}" env config vars set PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
        "${MGR}" env config vars set R_PROFILE_USER="$(pwd -P)/.Rprofile"
        "${MGR}" env config vars set RENV_PATHS_ROOT="$(pwd -P)/.renv"
    )
    (
        "${MGR}" activate ./conda
        Rscript -e "renv::restore()"
        Rscript -e "IRkernel::installspec(prefix='${CONDA_PREFIX}', rprofile='${R_PROFILE_USER}')"
    )
    (
        "${MGR}" activate ./conda
        flit install -s
    )
    (
        "${MGR}" activate ./conda
        pre-commit install
    )
elif [ "${1}" = "export" ]; then
    "${MGR}" env export -p ./conda --no-build | \
        yq -y 'del(.name, .prefix, .variables) | del(.dependencies[].pip?[] | select(startswith("cascade")))' \
        > conda.yaml
    (
        "${MGR}" activate ./conda
        Rscript -e "renv::snapshot()"
    )
elif [ "${1}" = "update" ]; then
    "${MGR}" env update -p ./conda -f conda.yaml
    (
        "${MGR}" activate ./conda
        Rscript -e "renv::restore()"
    )
else
    echo "Usage: ${0} <create|export|update>"
    exit 1
fi
