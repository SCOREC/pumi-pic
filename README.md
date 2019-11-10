## PUMI-PIC

CDash: https://my.cdash.org/index.php?project=pumi-pic

### Dependencies

- Kokkos https://github.com/kokkos/kokkos
- Omega_h https://github.com/SNLComputation/omega_h
- particle_structures https://github.com/SCOREC/particle_structures

### Setup

```
git clone git@github.com:SCOREC/pumi-pic.git
```

Developers and users who want to run `ctest` will also need to initialize
submodules for test data.

```
git submodule init
git submodule update
```

### Build on SCOREC RHEL7

```
mkdir build
cd !$
source ../pumi-pic/envRhel7Serial.sh
../pumi-pic/doConfig.sh ../pumi-pic /path/to/particle_structures/install/dir
make
```

### Build on SCOREC Blockade for Cuda

```
mkdir build
cd !$
source ../pumi-pic/envBlockadeCuda.sh
../pumi-pic/doConfig.sh ../pumi-pic /path/to/particle_structures/cuda_install/dir
make
```

### Run

```
cd build
ctest
```
