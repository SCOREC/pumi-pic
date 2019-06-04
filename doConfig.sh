#!/bin/bash
[ $# -ne 2 ] && echo "Usage: $0 <path to source> <path to particle structures
install >" && return 1
src=$1
ps=$2

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$ps

cmake $src \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_BUILD_TYPE=DEBUG \
      -DIS_TESTING=ON \
      -DTEST_DATA_DIR=$src/pumipic-data
