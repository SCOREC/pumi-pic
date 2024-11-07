export root=$PWD
module load cmake

function getname() {
  name=$1
  machine=perlmutter
  buildSuffix=${machine}-cuda
  echo "build-${name}"
}
export engpar=$root/`getname engpar`/install # This is where engpar will be (or is) installed
export kk=$root/`getname kokkos`/install   # This is where kokkos will be (or is) installed
export oh=$root/`getname omegah`/install  # This is where omega_h will be (or is) installed
export cab=$root/`getname cabana`/install # This is where cabana will be (or is) installed
export pumipic=$root/`getname pumi-pic`/install # This is where PumiPIC will be (or is) installed
export CMAKE_PREFIX_PATH=$engpar:$kk:$kk/lib64/cmake:$oh:$cab:$pumipic:$CMAKE_PREFIX_PATH
export MPICH_CXX=$root/kokkos/bin/nvcc_wrapper