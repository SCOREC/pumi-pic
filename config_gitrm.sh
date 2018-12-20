#!/bin/bash

OMEGA_H=/home/gp/Programs/omega_h_install
KOKKOS=/home/gp/Programs/kokkos_install

#-DCMAKE_CXX_FLAGS="-std=c++11 -Wall -Wextra" \

export CMAKE_PREFIX_PATH=$OMEGA_H:$CMAKE_PREFIX_PATH

cmake \
 -DOMEGA_H_PREFIX=$OMEGA_H \
 -DCMAKE_BUILD_TYPE=Debug \
 $1

