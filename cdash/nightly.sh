#!/bin/bash -x
(
#cdash output root
d=/users/d_zxg06726/nightlyBuilds/pumipic_build
exec > $d/nightly_log.txt 2>&1

source /etc/profile
# source /users/d_zxg06726/.bash_profile

#setup lmod
export PATH=/usr/share/lmod/lmod/libexec:$PATH

#setup spack modules
unset MODULEPATH

module use /opt/scorec/spack/rhel9/v0222_2/lmod/linux-rhel9-x86_64/Core/
module load gcc/13.2.0-4eahhas
module load mpich/4.2.3-62uy3hd
module load cuda/12.6.2-gqq65nw
module load cmake

cd $d
#remove compilation directories created by previous nightly.cmake runs
[ -d build ] && rm -rf build/

#install kokkos
[ ! -d kokkos ] && git clone --branch 5.0.0 https://github.com/kokkos/kokkos.git
[ -d build-kokkos ] && rm -rf build-kokkos
cmake -S kokkos -B build-kokkos \
  -DCMAKE_CXX_COMPILER=$d/kokkos/bin/nvcc_wrapper \
  -DCMAKE_INSTALL_PREFIX=build-kokkos/install \
  -DKokkos_ARCH_TURING75=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_IMPL_VIEW_LEGACY=ON \
  -DKokkos_ENABLE_DEBUG=ON
cmake --build build-kokkos -j 4 --target install

#install omega_h
[ ! -d omega_h ] && git clone https://github.com/SCOREC/omega_h.git
cd omega_h && git pull && cd -
[ -d build-omega_h ] && rm -rf build-omega_h
cmake -S omega_h -B build-omega_h \
  -DCMAKE_INSTALL_PREFIX=build-omega_h/install \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_BUILD_TYPE=debug \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=ON \
  -DOmega_h_CUDA_ARCH=75 \
  -DOmega_h_USE_MPI=ON \
  -DBUILD_TESTING=ON \
  -DKokkos_PREFIX=$d/build-kokkos/install/lib64/cmake
cmake --build build-omega_h -j 4 --target install

#install EnGPar
[ ! -d EnGPar ] && git clone https://github.com/SCOREC/EnGPar.git
cd EnGPar && git pull && cd -
[ -d build-EnGPar ] && rm -rf build-EnGPar
cmake -S EnGPar -B build-EnGPar \
  -DCMAKE_INSTALL_PREFIX=build-EnGPar/install \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DENABLE_PARMETIS=OFF \
  -DENABLE_PUMI=OFF \
  -DIS_TESTING=OFF
cmake --build build-EnGPar -j 4 --target install

#install Cabana
[ ! -d Cabana ] && git clone https://github.com/ECP-copa/Cabana.git
cd Cabana && git pull && cd -
[ -d build-Cabana ] && rm -rf build-Cabana
cmake -S Cabana -B build-Cabana \
  -DCMAKE_INSTALL_PREFIX=build-Cabana/install \
  -DKokkos_DIR=$d/build-kokkos/install/lib64/cmake/Kokkos \
  -DCMAKE_BUILD_TYPE="Release" \
  -DCMAKE_CXX_COMPILER=$d/kokkos/bin/nvcc_wrapper \
  -DCabana_ENABLE_TESTING=OFF \
  -DCabana_ENABLE_EXAMPLES=OFF
cmake --build build-Cabana -j 4 --target install

#download testcases
[ ! -d pumipic-data ] && git clone https://github.com/SCOREC/pumipic-data.git
cd pumipic-data && git pull && cd -

touch $d/startedCoreNightly
#run nightly.cmake script
ctest -V --script $d/nightly.cmake
touch $d/doneCoreNightly
)
