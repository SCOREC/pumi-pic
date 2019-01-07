#!/bin/bash
[ $# -ne 1 ] && echo "Usage: $0 <path to source>" && exit 1
src=$1

cmake $src \
-DIS_TESTING=ON \
-DTEST_DATA_DIR=$src/pumipic-data \
-DCMAKE_INSTALL_PREFIX=$PWD/install
