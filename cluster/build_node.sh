#!/bin/bash
###########################################
# Helper script to build clgen across nodes
# Run `scl enable devtoolset-7 bash'
# before executing it.
###########################################

mkdir -p /disk/scratch/s1879742/clgen_build
cd /disk/scratch/s1879742/clgen_build

cmake ~/clgen

make -j 16
make -j 16 install_lib_ap
make -j 16 install_lib_ic
make -j 16 Grewe
make -j 16 clang_rewriter