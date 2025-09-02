#!/bin/bash

set -e

singularity build --fakeroot anndata.sif anndata.def
singularity build --fakeroot causalicp.sif causalicp.def
singularity build --fakeroot --nv causaldag.sif causaldag.def
singularity build --fakeroot --nv cdt.sif cdt.def
singularity build --fakeroot --nv notears.sif notears.def
singularity build --fakeroot --nv nobears.sif nobears.def
singularity build --fakeroot --nv dcdi.sif dcdi.def
singularity build --fakeroot --nv dcdfg.sif dcdfg.def
singularity build --fakeroot --nv dagma.sif dagma.def
singularity build --fakeroot --nv biolord.sif biolord.def
singularity build --fakeroot --nv cpa.sif cpa.def
singularity build --fakeroot --nv gears.sif gears.def
singularity build --fakeroot --nv scgpt.sif scgpt.def
singularity build --fakeroot --nv scfoundation.sif scfoundation.def
