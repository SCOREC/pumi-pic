# particle_structures
Sell-C-sigma with vertical slicing for unstructured mesh particle-in-cell (PIC) 

# Dependencies

- Kokkos
- Thrust (on GPUs)

# Building on SCOREC RHEL7

## Kokkos with OpenMP

```
module load gcc/7.3.0-bt47fwr mpich/3.2.1-niuhmad cmake/3.13.1-ovasnmm trilinos/develop-debug-openmp-ackkufk
cmake /path/to/particle_structures/source/dir -DENABLE_KOKKOS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install -DCMAKE_CXX_COMPILER=mpicxx
make
make install
```

## Kokkos with Cuda

```
module load gcc/7.3.0-bt47fwr mpich/3.3-diz4f6i cmake/3.13.1-ovasnmm
kk=/lore/cwsmith/develop/build-kokkos-blockade-cuda/install
omega_h=/lore/cwsmith/develop/build-omegah-rhel7-cuda-latest/install
export CMAKE_PREFIX_PATH=$kk:$omega_h:$CMAKE_PREFIX_PATH
export MPICH_CXX=/lore/cwsmith/develop/kokkos/bin/nvcc_wrapper
cuda=/usr/local/cuda-10.1
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$LD_LIBRARY_PATH
cmake /path/to/particle_structures/source/dir -DENABLE_KOKKOS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install-cuda -DCMAKE_CXX_COMPILER=mpicxx
make
make install
```

Debug symbols can be added by appending `-DCMAKE_BUILD_TYPE=DEBUG` to the cmake
command.

# Running tests

Running `ctest` will execute a several unit tests

