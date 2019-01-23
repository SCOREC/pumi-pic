# particle_structures
sandbox to test performance of particle data structures

CDash https://my.cdash.org/index.php?project=particle_structures

# Goals and design

We want to understand the performance of different data structures that store particles for a PIC code.  The performance test will entail:
1) loading particles into the data structure following a gaussian, exponential, or uniform distribution across mesh elements
2) setting an initial position for the particles that is != 0 and != 1 - they can all be the same
3) computing a new position for each particle using a random direction and fixed distance
4) storing the new position in a second particle data structure

The inputs to the test are:
 - number of mesh elements
 - number of particles
 - distribution of particles
 - number of push operations to perform (steps 3 and 4 above)

There will not be any mesh, particles won't leave their initial 'element' in the data structure, and the host code will only run in serial.

Kokkos will be used for GPU parallelization of the push  operation (steps 3 and 4 above).

## Data structures to compare

### Sell-C-sigma with vertical slicing

See the discussion in the hybridpic working document.

### Flat arrays

Testing on flat arrays will establish an upper bound on performance.

One contiguous array is created for each dimension; 'x','y','z'.

There is no association of particles to elements.  During initialization we will ignore the 'distribution' input.


## Gather results

```
    grep "kokkos array push (seconds)" d*.log | awk '{print $1 "," $5}' > arrayPush.csv
    grep "kokkos scs push (seconds)" d*.log | awk '{print $1 "," $5}' > scsPush.csv
    paste arrayPush.csv scsPush.csv > push.csv
    sed -i s/.log:kokkos//g push.csv
    sed -i s/_sorted//g push.csv
    sed -i s/_/,/g push.csv
    sed -i s/[depCV]//g push.csv
    tr '\t' "," < push.csv > push2.csv
    echo 'd,e,p,C,V,array,d,e,p,C,V,scs' > headers.csv
    cat headers.csv push2.csv > push.csv
    rm push2.csv
```

# Dependencies

- Kokkos

# Building on SCOREC RHEL7

## Kokkos with OpenMP

```
module load gcc/7.3.0-bt47fwr mpich/3.2.1-niuhmad cmake/3.13.1-ovasnmm trilinos/develop-debug-openmp-ackkufk
cmake /path/to/particle_structures/source/dir -DENABLE_KOKKOS=ON
make
make install
```

Debug symbols can be added by appending `-DCMAKE_BUILD_TYPE=DEBUG` to the cmake
command.

# Running tests

Running `ctest` will execute a several implementations of a pseudo push:
the implementations include serial+arrays, kokkos+arrays, serial+SCS,
Kokkos+SCS, and Kokkos+SCS+macros.

