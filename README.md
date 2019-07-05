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

### Test Description

Search test routines : test_adj, test_collision

Usage: ./gitrm mesh <init> <final>
Example: ./gitrm cube.msh 2,0.5,0.2  4,0.9,0.3

The linked pumipic-data, has sample cube.msh, which is a gmsh mesh of a rectangular block !
The dimensions of the block is x:0 to 10m; y:0 to 1m; z:0 to 1m. In the above example, start and final positions are within the domain. If the destination is outside the domain (eg: 4, -1, 0.3), the collision routine will run and intersection point will be output. 
