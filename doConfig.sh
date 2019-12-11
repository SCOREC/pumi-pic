#!/bin/bash
[ $# -ne 1 ] && echo "Usage: $0 <path to source>" && return 1
src=$1

ncxx=$MY_NCXX
cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
      -DCUDA_NVCC_FLAGS:STRING="-G -Xcompiler -rdynamic" \
      -DNetCDF_PREFIX:PATH=$ncxx \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_BUILD_TYPE=Release \
      -DIS_TESTING=ON \
      -DTEST_DATA_DIR=$src/pumipic-data \
      $src

