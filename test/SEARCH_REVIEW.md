# Fuad search integration review

Branch: `review/fuad-search-and-tests`, based on upstream `master` at `6dfdde2`.
Validation date: November 16, 2026.

## Imported implementation

The 18 unmerged commits leading to
[`make_search_class` at `6a9f5d8`](https://github.com/Fuad-HH/pumi-pic/tree/6a9f5d87cc058c28f3fcf8a4baea0fff197896d9)
were cherry-picked in order, followed by
[`edge_case_fix` at `ace8c4c`](https://github.com/Fuad-HH/pumi-pic/commit/ace8c4c46fc734e73b6816f704a8abc226049e4b).
This includes optional/cached tolerance, cached element measures, standardized
tracing state and callback advancement, line-segment tracing, `ParticleTracer`,
and the shared-face/near-edge recovery changes. The earlier generic callback
implementation and ray/segment helpers were already on upstream `master`.

No additional search algorithm fixes have been made during integration. The
failures below remain visible for review.

## Consolidated test sources

The latest versions of the earlier tests come from
[`add_functor_to_search_mesh` at `961a919`](https://github.com/Fuad-HH/pumi-pic/tree/961a919565c71b27744a50372ae03d5836fe3c82).
That branch includes the earlier `line_edge_2d_test` and `find_exit_face_test`
work. Superseded revisions of the same scenarios are not duplicated.

| Source | Local executable(s) |
| --- | --- |
| `make_search_class`: `test_trace_particle_through_mesh.cpp` | `test_trace_particle_through_mesh` |
| `make_search_class`: `test_search_mesh.cpp` | `test_search_mesh` |
| `make_search_class`: `test_ParticleTracer.cpp` | `test_ParticleTracer` |
| Earlier branch: `test_find_exit_face.cpp` | `test_find_exit_face`, including a separate `--bcc` invocation |
| Earlier branch: `test_find_exit_face_3D.cpp` | `test_find_exit_face_3D`, including a separate `--bcc` invocation |
| Earlier branch: `test_line_edge_2d.cpp` | `test_line_edge_2d` |
| Earlier branch: `moller_trumbore_line_tri_test.cpp` | `test_triangle_intersections`; existing upstream `moller_trumbore_test` retained |
| Earlier branch: `test_search_mesh.cpp` | `test_search_mesh_gmsh`; retained alongside the newer generated-mesh test |
| Earlier branch: `test_particle_search.cpp` | `test_particle_search` |
| Existing upstream randomized tests | `test_adj_2d`, `test_adj_3d`; restored to CTest |

Test integration adapts removed APIs (`particle_search`, `set_new_element`, and
the old segment flag), updates the destination/exit-face expectations to the
line-segment contract, and fixes checks that previously did not affect the
exit status. The original triangle segment checks now contribute to test success,
and the third 2D trajectory's coordinate check is enforced. The 3D intersection
check compares the correct expected z coordinate. Duplicate declarations in
the non-Cabana paths and GNU `typeof` usages in the newer tests were corrected.
The existing adjacency validator's call was updated for the new signature and
correct tolerance/debug argument order.

All 13 consolidated tracing/adjacency tests have the CTest label `search` and a
120-second timeout. The previously commented BCC diagnostics run as ordinary
tests; failures are not inverted or disabled.

## Build and validation

The [Spack development setup](../spack/README.md) builds this checkout with
GCC 12.3, MPICH 4.1.1, Kokkos 4.7.04 (Serial/OpenMP), Omega_h 11.4.0-scorec,
Cabana 0.7.0, and EnGPar. Both `spack install` and the complete Debug test build
succeeded. Tests ran with two OpenMP threads. CUDA has not been validated.

The full Debug run completed: **68 of 74 tests passed**. Of the 13 tests labeled
`search`, eight passed and five failed. The existing ray/segment test, 2D search
integration test, and all three pseudo-push/search cases also passed.

| Failing test | Finding |
| --- | --- |
| `test_adj_2d`, `test_adj_3d` | Regression: both pass on upstream `6dfdde2` with the same dependencies. The imported `check_model_intersection` writes `xFace[pid]` even when `requireIntersection=false` and the face array is empty. GDB identifies the write at `pumipic_adjacency.tpp:467`. |
| `test_find_exit_face` | The clockwise 2D Gmsh fixture has signed area -25. The raw intersection search reports no exit face; the standalone edge test explicitly compensates for orientation and passes. Mesh orientation handling needs review. |
| `test_find_exit_face_bcc` | Two origins with the same destination select the same edge, although their trajectories cross different edges; this is the diagnostic from PR #128. |
| `test_find_exit_face_3D_bcc` | The barycentric path selects face 55 for both trajectories instead of faces 56 and 55. |
| `smoke_test_particle` | Also fails on upstream `6dfdde2` in the identical environment, reporting particle-structure initialization errors. This failure is not introduced by the cherry-picks. |

The upstream comparison used a separate source snapshot under `/tmp`; it did not
modify the review branch. Local detailed output is in
`build/debug/Testing/Temporary/LastTest.log`, with baseline and GDB logs copied
into `build/debug/review-logs/`.

The newest particle-loss commit carries author-reported million-particle
validation, but adds no dedicated regression tests. The imported suite therefore
does not independently establish that result. Further review should also cover
`ParticleTracer`'s fixed particle-member indices, cached state after migration or
mesh changes, and the unbounded tracing loop.
