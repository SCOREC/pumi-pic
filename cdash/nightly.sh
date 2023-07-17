#!/bin/bash

#load system modules
source /etc/profile.d/modules.sh
source /etc/profile

export root=/lore/castia5/nightlyBuilds/pumipic

module use /opt/scorec/spack/dev/lmod/linux-rhel7-x86_64/Core
module unuse /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core
module load gcc/7.4.0-c5aaloy cuda/11.4
module load mpich/3.3.1-bfezl2l
module load cmake

function getname() {
  name=$1
  buildSuffix=cranium-cuda114
  echo "build-${name}-${buildSuffix}"
}
export engpar=$root/`getname engpar`/install # This is where engpar will be (or is) installed
export kk=$root/`getname kokkos`/install   # This is where kokkos will be (or is) installed
export oh=$root/`getname omegah`/install  # This is where omega_h will be (or is) installed
export oh1050=$root/`getname omegah1050`/install
export pumipic=$root/`getname pumipic`/install # This is where PumiPIC will be (or is) installed
export CMAKE_PREFIX_PATH=$engpar:$kk:$oh:$pumipic:$CMAKE_PREFIX_PATH
export MPICH_CXX=$root/kokkos/bin/nvcc_wrapper

cd $root
[ ! -d kokkos ] && git clone -b 3.4.01 git@github.com:kokkos/kokkos.git
[ -d $kk ] && rm -rf ${kk%%install}
cmake -S kokkos -B ${kk%%install} \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DKokkos_ARCH_TURING75=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=off \
  -DKokkos_ENABLE_CUDA=on \
  -DKokkos_ENABLE_CUDA_LAMBDA=on \
  -DKokkos_ENABLE_DEBUG=on \
  -DCMAKE_INSTALL_PREFIX=$kk
cmake --build ${kk%%install} --target install -j 24

cd $root
[ ! -d EnGPar] && git clone git@github.com:SCOREC/EnGPar.git
cd EnGPar && git pull && cd -
[ -d $engpar ] && rm -rf ${engpar%%install}
cmake -S EnGPar -B ${engpar%%install} \
  -DCMAKE_INSTALL_PREFIX=$engpar \
  -DCMAKE_C_COMPILER="mpicc" \
  -DCMAKE_CXX_COMPILER="mpicxx" \
  -DCMAKE_CXX_FLAGS="-std=c++11" \
  -DENABLE_PARMETIS=OFF \
  -DENABLE_PUMI=OFF \
  -DIS_TESTING=OFF
cmake --build ${engpar%%install} --target install -j8 

cd $root
[ ! -d omega_h ] && git clone git@github.com:SCOREC/omega_h.git
cd omega_h && git checkout master && git pull && cd -
[ -d $oh ] && rm -rf ${oh%%install}
cmake -S omega_h -B ${oh%%install} \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_BUILD_TYPE=debug \
  -DCMAKE_INSTALL_PREFIX=$oh \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=on \
  -DOmega_h_CUDA_ARCH=75 \
  -DOmega_h_USE_MPI=on  \
  -DBUILD_TESTING=on  \
  -DKokkos_PREFIX=$kk/lib64/cmake
cmake --build ${oh%%install} --target install -j8

cd omega_h && git checkout scorec-v10.5.0 && cd -
[ -d $oh1050 ] && rm -rf ${oh1050%%install}
cmake -S omega_h -B ${oh1050%%install} \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_BUILD_TYPE=debug \
  -DCMAKE_INSTALL_PREFIX=$oh1050 \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=on \
  -DOmega_h_CUDA_ARCH=75 \
  -DOmega_h_USE_MPI=on  \
  -DBUILD_TESTING=on  \
  -DKokkos_PREFIX=$kk/lib64/cmake
cmake --build ${oh1050%%install} --target install -j8

set +e
set +x

d=/lore/castia5/nightlyBuilds/pumipic
cd $d
#remove old compilation
[ -d build_pumipic ] && rm -rf build_pumipic/
cd $d/repos/pumipic
git pull
git submodule init
git submodule update

mpicxx -show
#run nightly test script
ctest -VV -D Nightly -S $d/repos/pumipic/cdash/nightly.cmake
