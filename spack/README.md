# PUMI-PiC development environment

This environment follows the PCMS development setup: Spack manages dependencies,
and `develop` points at this checkout. It enables Cabana and Kokkos Serial/OpenMP
with a Debug build. The SCOREC mirror can supply cached dependencies.

From the repository root, with Spack available in your shell:

```bash
spack env activate ./spack
spack concretize
spack install
git submodule update --init pumipic-data
bash spack/build.sh
```

The package recipe builds the library. `build.sh` additionally enables the library
and particle-structure tests in `build/debug`, using the same concrete dependency
environment. Repeat that command after editing the source.

Run tests inside the build environment so MPI and shared libraries are available:

```bash
OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
  spack -e ./spack build-env pumi-pic -- \
  ctest --test-dir build/debug --output-on-failure
```

Set `PUMIPIC_BUILD_JOBS`, `PUMIPIC_BUILD_DIR`, or `PUMIPIC_TEST_DATA_DIR` to override
the default four build jobs, build directory, or test-data location. The generated
lockfile and view are local artifacts; the checked-in manifest records the setup.
