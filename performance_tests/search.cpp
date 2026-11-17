#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>
#include <pumipic_adjacency.hpp>
#include <team_policy.hpp>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace o = Omega_h;
namespace p = pumipic;
using Particle = p::MemberTypes<p::Vector3d, p::Vector3d, int>;
using PS = p::ParticleStructure<Particle>;
using Exec = Kokkos::DefaultExecutionSpace;

int main(int argc, char** argv) {
  o::Library library(&argc, &argv);
  if (argc < 5 || argc > 6) {
    std::fprintf(stderr, "Usage: %s <2|3> <particles> <repetitions> <no-output|segment> [nonconvex]\n", argv[0]);
    return 1;
  }
  const int dim = std::stoi(argv[1]);
  const int count = std::stoi(argv[2]);
  const int repetitions = std::stoi(argv[3]);
  const std::string mode = argv[4];
  const bool segment = mode == "segment";
  const bool nonconvex = argc == 6 && std::string(argv[5]) == "nonconvex";
  if ((dim != 2 && dim != 3) || count <= 0 || repetitions <= 0 ||
      (mode != "no-output" && !segment) || (nonconvex && dim != 2)) return 1;
  OMEGA_H_CHECK(library.world()->size() == 1);

  o::Mesh mesh(&library);
  if (nonconvex) {
    // L domain: [0,2] x [0,1] union [0,1] x [1,2]. The complete segment
    // (0.5,1.25) -> (1.8,0.3) stays inside, but minimum-BCC walking can
    // choose the reentrant boundary at x=1 and incorrectly delete the particle.
    mesh.set_comm(library.self());
    o::build_from_elems_and_coords(&mesh, OMEGA_H_SIMPLEX, 2,
        o::LOs{0,1,4, 0,4,3, 1,2,5, 1,5,4, 3,4,7, 3,7,6},
        o::Reals{0,0, 1,0, 2,0, 0,1, 1,1, 2,1, 0,2, 1,2});
  } else {
    mesh = o::build_box(library.self(), OMEGA_H_SIMPLEX, 1, 1,
                        dim == 3 ? 1 : 0, 20, 20, dim == 3 ? 20 : 0);
  }
  const auto vertices = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto nelems = mesh.nelems();
  PS::kkLidView ppe("particles_per_element", nelems);
  PS::kkGidView gids("element_ids", nelems);
  o::parallel_for(nelems, OMEGA_H_LAMBDA(int e) {
    gids(e) = e;
    if (nonconvex) {
      const auto verts = o::gather_verts<3>(vertices, e);
      const auto bcc = o::barycentric_from_global<2, 2>(o::Vector<2>{0.5, 1.25},
          o::gather_vectors<3, 2>(coords, verts));
      ppe(e) = p::all_positive(bcc, 0) ? count : 0;
    } else {
      ppe(e) = count / nelems + (e < count % nelems);
    }
  });
  auto policy = p::TeamPolicyAuto(nelems, 32);
  std::unique_ptr<PS> particles(new p::DPS<Particle>(policy, nelems, count, ppe, gids));
  auto origin = particles->get<0>();
  auto target = particles->get<1>();
  auto ids = particles->get<2>();
  auto initialize = PS_LAMBDA(int e, int pid, bool mask) {
    if (!mask) return;
    ids(pid) = pid;
    if (nonconvex) {
      origin(pid,0) = 0.5; origin(pid,1) = 1.25; origin(pid,2) = 0;
      target(pid,0) = 1.8; target(pid,1) = 0.3; target(pid,2) = 0;
    } else {
      o::Vector<3> center = o::zero_vector<3>();
      if (dim == 2) {
        const auto verts = o::gather_verts<3>(vertices, e);
        const auto points = o::gather_vectors<3, 2>(coords, verts);
        for (int v = 0; v < 3; ++v)
          for (int d = 0; d < 2; ++d) center[d] += points[v][d] / 3;
      } else {
        const auto verts = o::gather_verts<4>(vertices, e);
        const auto points = o::gather_vectors<4, 3>(coords, verts);
        for (int v = 0; v < 4; ++v)
          for (int d = 0; d < 3; ++d) center[d] += points[v][d] / 4;
      }
      for (int d = 0; d < 3; ++d) {
        origin(pid,d) = center[d];
        target(pid,d) = d < dim ? 0.9 * center[d] + 0.049 + d * 0.0007 : 0;
      }
    }
  };
  p::parallel_for(particles.get(), initialize, "initialize_search_benchmark");

  std::vector<double> times;
  o::Write<o::LO> parents, faces;
  o::Write<o::Real> points;
  // Include allocations and search setup in timing; exclude particle creation,
  // compilation, warm-up, output, and independent result validation.
  for (int iteration = -2; iteration < repetitions; ++iteration) {
    parents = o::Write<o::LO>();
    faces = o::Write<o::LO>();
    points = o::Write<o::Real>();
    Kokkos::fence();
    Kokkos::Timer timer;
    const bool found = p::search_mesh(mesh, particles.get(), origin, target, ids,
                                     parents, segment, faces, points, 200);
    Kokkos::fence();
    const double seconds = timer.seconds();
    if (!found) return 2;
    if (iteration >= 0) times.push_back(seconds);
  }
  o::Write<o::LO> invalid(1, 0), active(1, 0);
  auto validate = PS_LAMBDA(int, int pid, bool mask) {
    if (!mask) return;
    Kokkos::atomic_add(&active[0], 1);
    bool inside = false;
    if (parents[pid] >= 0 && parents[pid] < nelems) {
      if (dim == 2) {
        const auto verts = o::gather_verts<3>(vertices, parents[pid]);
        inside = p::all_positive(o::barycentric_from_global<2, 2>(
            p::makeVector2(pid, target), o::gather_vectors<3, 2>(coords, verts)), 1e-8);
      } else {
        const auto verts = o::gather_verts<4>(vertices, parents[pid]);
        inside = p::all_positive(o::barycentric_from_global<3, 3>(
            p::makeVector3(pid, target), o::gather_vectors<4, 3>(coords, verts)), 1e-8);
      }
    }
    if (!inside) Kokkos::atomic_add(&invalid[0], 1);
  };
  p::parallel_for(particles.get(), validate, "validate_search_benchmark");
  if (nonconvex) {
    const auto exposed = o::mark_exposed_sides(&mesh);
    const auto down = mesh.ask_down(2, 1);
    const auto up = mesh.ask_up(1, 2);
    const auto areas = measure_elements_real(&mesh);
    auto check_point = PS_LAMBDA(int initial_parent, int pid, bool mask) {
      if (!mask) return;
      o::LO loops;
      const auto located = p::search_mesh_2d_pt(exposed, vertices, coords, down,
          up, areas, p::makeVector2(pid, origin), p::makeVector2(pid, target),
          pid, initial_parent, loops, 200);
      if (located < 0 || located != parents[pid]) Kokkos::atomic_add(&invalid[0], 1);
    };
    p::parallel_for(particles.get(), check_point, "validate_nonconvex_device_search");
  }
  const int errors = o::HostRead<o::LO>(invalid)[0];
  OMEGA_H_CHECK(o::HostRead<o::LO>(active)[0] == count);
  std::sort(times.begin(), times.end());
  const double median = times[times.size()/2];
  std::printf("SEARCH_RESULT backend=%s dimension=%d mode=%s particles=%d repetitions=%d "
              "nonconvex=%d median_ms=%.6f particles_per_second=%.0f invalid=%d\n",
              Exec::name(), dim, mode.c_str(), count, repetitions, nonconvex,
              median * 1000, count / median, errors);
  return errors == 0 ? 0 : 2;
}
