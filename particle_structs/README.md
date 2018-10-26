# particle_structures
sandbox to test performance of particle data structures

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

### flat arrays

Testing on flat arrays will establish an upper bound on performance.

One contiguous array is created for each dimension; 'x','y','z'.

There is no association of particles to elements.  During initialization we will ignore the 'distribution' input.


## Gather results

```
  grep 'kokkos array push (seconds)' d*.log | awk '{print $1 "," $5}' > arrayPush.csv
  grep 'kokkos scs push (seconds)' d*.log | awk '{print $1 "," $5}' > scsPush.csv
  paste arrayPush.csv scsPush.csv > push.csv
  sed -i s/.log:kokkos//g push.csv
  sed -i s/_/,/g push.csv
  sed -i s/[dep]//g push.csv
  tr '\t' "," < push.csv > push2.csv
  echo 'd,e,p,array,d,e,p,scs' > headers.csv
  cat headers.csv push2.csv > push.csv
  rm push2.csv
```
