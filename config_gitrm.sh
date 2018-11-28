#!/bin/bash

OMEGA_H=/home/gp/Programs/omega_h_install/
KOKKOS=/home/gp/Programs/kokkos_install

cmake \
-DCMAKE_PREFIX_PATH=$OMEGA_H \
-DOMEGA_H_PREFIX=$OMEGA_H \
-DOmega_h_ENABLE_Kokkos=ON \
-DKokkos_PREFIX=$KOKKOS \
-DCMAKE_BUILD_TYPE=Debug \
`dirname "$(readlink -f "$0")"`

#-DCMAKE_CXX_FLAGS="-std=c++11 -Wall -Wextra" \
