#!/bin/bash
[ $# -ne 2 ] && echo "Usage: $0 <path to source> <path to particle structures
install >" && return 1
src=$1
ps=$2

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$ps

cmake $src \
-DIS_TESTING=ON \
-DTEST_DATA_DIR=$src/pumipic-data \
-DCMAKE_INSTALL_PREFIX=$PWD/install
