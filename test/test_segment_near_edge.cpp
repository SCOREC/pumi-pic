#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <pumipic_adjacency.hpp>
#include <team_policy.hpp>
#include <climits>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

namespace o = Omega_h;
namespace p = pumipic;
using Particle = p::MemberTypes<p::Vector3d, p::Vector3d, int, p::Vector3d>;
using PS = p::ParticleStructure<Particle>;

// Host-side reference check, independent of the tracing/recovery helpers.
o::Vector<4> barycentric(const o::HostRead<o::LO>& vertices,
                         const o::HostRead<o::Real>& coordinates,
                         int element, const o::Vector<3>& point) {
  o::Matrix<3, 4> tetrahedron;
  for (int v = 0; v < 4; ++v)
    for (int d = 0; d < 3; ++d)
      tetrahedron[v][d] = coordinates[3 * vertices[4 * element + v] + d];
  return o::barycentric_from_global<3, 3>(point, tetrahedron);
}

struct RecordWalk {
  bool verbose;
  int slot;
  o::Vector<3> start;
  o::Vector<3> target_point;
  o::Real tolerance;
  mutable int previous_face = -1;

  void trace_faces(o::Mesh& mesh, int parent) const {
    const o::HostRead<o::LO> vertices(mesh.ask_elem_verts());
    const o::HostRead<o::LO> sides(mesh.ask_down(3, 2).ab2b);
    const o::HostRead<o::LO> face_vertices(mesh.ask_verts_of(2));
    const o::HostRead<o::Real> coordinates(mesh.coords());
    const auto up = mesh.ask_up(2, 3);
    const o::HostRead<o::LO> offsets(up.a2ab);
    const o::HostRead<o::LO> adjacent(up.ab2b);
    o::Few<o::LO, 4> tet_vertices;
    for (int v = 0; v < 4; ++v) tet_vertices[v] = vertices[4*parent+v];
    const auto direction = (target_point - start) / o::norm(target_point - start);
    const auto start_bcc = barycentric(vertices, coordinates, parent, start);
    const auto end_bcc = barycentric(vertices, coordinates, parent, target_point);
    std::printf("  TET start_bcc=(%.17g, %.17g, %.17g, %.17g) end_bcc=(%.17g, %.17g, %.17g, %.17g)\n",
        start_bcc[0], start_bcc[1], start_bcc[2], start_bcc[3],
        end_bcc[0], end_bcc[1], end_bcc[2], end_bcc[3]);
    for (int local = 0; local < 4; ++local) {
      const int face_id = sides[4*parent+local];
      int neighbor = -1;
      for (int i = offsets[face_id]; i < offsets[face_id+1]; ++i)
        if (adjacent[i] != parent) neighbor = adjacent[i];
      o::Few<o::LO, 3> fv;
      o::Few<o::Vector<3>, 3> face;
      for (int v = 0; v < 3; ++v) {
        fv[v] = face_vertices[3*face_id+v];
        for (int d = 0; d < 3; ++d) face[v][d] = coordinates[3*fv[v]+d];
      }
      const int flip = p::isFaceFlipped(local, fv, tet_vertices);
      o::Vector<3> crossing;
      o::Real projection, closeness, parameter;
      const bool accepted = p::line_segment_intersects_triangle(face, start,
          target_point, crossing, tolerance, flip, projection, closeness, parameter);
      const auto edge1 = face[2-flip] - face[0];
      const auto edge2 = face[flip+1] - face[0];
      const auto offset = start - face[0];
      const double u = o::inner_product(offset, o::cross(direction, edge2)) / projection;
      const double v = o::inner_product(direction, o::cross(offset, edge1)) / projection;
      o::Vector<3> face_bcc;
      p::find_barycentric_tri_simple(face, crossing, face_bcc);
      const double interiority = std::min(std::min(face_bcc[0], face_bcc[1]), face_bcc[2]);
      std::printf("  FACE local=%d id=%d neighbor=%d vertices=(%d,%d,%d) previous=%d accepted=%d dproj=%.17g t=%.17g uvw=(%.17g,%.17g,%.17g) interiority=%.17g\n",
          local, face_id, neighbor, fv[0], fv[1], fv[2], face_id == previous_face,
          accepted, projection, parameter, u, v, 1-u-v, interiority);
    }
  }

  template <typename Origin, typename Target>
  void operator()(o::Mesh& mesh, PS* particles, o::Write<o::LO>& elements,
                  o::Write<o::LO>& next, o::Write<o::LO>& faces,
                  o::Write<o::LO>& exits, o::Write<o::Real>& points,
                  o::Write<o::LO>& done, Origin origin, Target target) const {
    if (verbose) {
      const auto parent = o::HostRead<o::LO>(elements)[slot];
      const auto next_parent = o::HostRead<o::LO>(next)[slot];
      const auto face = o::HostRead<o::LO>(exits)[slot];
      const auto crossing = o::HostRead<o::Real>(points);
      std::printf("WALK parent=%d face=%d next=%d point=(%.17g, %.17g, %.17g)\n",
                  parent, face, next_parent, crossing[3*slot],
                  crossing[3*slot+1], crossing[3*slot+2]);
      trace_faces(mesh, parent);
      previous_face = face;
    }
    p::RemoveParticleOnGeometricModelExit<Particle, Origin> advance(mesh, true);
    advance(mesh, particles, elements, next, faces, exits, points, done, origin, target);
  }
};

int main(int argc, char** argv) {
  o::Library library(&argc, &argv);
  if (argc != 2 && argc != 3) {
    std::fprintf(stderr, "Usage: %s <cube/7k.osh> [--trace]\n", argv[0]);
    return 2;
  }
  const bool verbose = argc == 3 && std::string(argv[2]) == "--trace";
  auto mesh = o::read_mesh_file(argv[1], library.world());
  OMEGA_H_CHECK(mesh.dim() == 3 && library.world()->size() == 1);

  // Snapshot of particle 20359 from the million-particle, edge-initialized
  // CUDA test. Decimal literals preserve the captured double-precision bits.
  // This also fails with Fuad's imported search at 3c8f405, before the common
  // tracer consolidation: it returns element 329 instead of containing cell 373.
  const int initial_parent = 2303;
  const o::Vector<3> start{18.887680337564333, 38.93251706677065, 38.930368214412653};
  const o::Vector<3> target{18.88790248127583, 30.986506687317515, 30.986196754518581};
  const o::Real tolerance = 1e-8;
  const o::HostRead<o::LO> vertices(mesh.ask_elem_verts());
  const o::HostRead<o::Real> coordinates(mesh.coords());
  if (!p::all_positive(barycentric(vertices, coordinates, initial_parent, start), tolerance)) {
    std::fprintf(stderr, "Fixture mismatch: starting point is outside element 2303\n");
    return 2;
  }
  int reference_count = 0;
  for (int element = 0; element < mesh.nelems(); ++element) {
    if (p::all_positive(barycentric(vertices, coordinates, element, target), tolerance)) {
      std::printf("REFERENCE destination lies in element %d\n", element);
      ++reference_count;
    }
  }
  if (!reference_count) return 2;
  if (verbose) {
    // Independent reference: tetrahedral barycentric coordinates vary linearly
    // along a segment. Intersect their four nonnegative parameter intervals.
    for (int element = 0; element < mesh.nelems(); ++element) {
      const auto a = barycentric(vertices, coordinates, element, start);
      const auto b = barycentric(vertices, coordinates, element, target);
      double enter = 0, leave = 1;
      for (int v = 0; v < 4; ++v) {
        const double slope = b[v] - a[v];
        if (slope > 0) enter = std::max(enter, -a[v]/slope);
        else if (slope < 0) leave = std::min(leave, -a[v]/slope);
        else if (a[v] < 0) leave = -1;
      }
      if (enter <= leave) {
        std::printf("REFERENCE segment element=%d enter=%.17g leave=%.17g\n",
                    element, enter, leave);
        if (leave - enter > 1e-13) {
          RecordWalk reference{true, -1, start, target, tolerance};
          reference.trace_faces(mesh, element);
        }
      }
    }
  }

  PS::kkLidView ppe("particles_per_element", mesh.nelems());
  PS::kkGidView gids("element_ids", mesh.nelems());
  o::parallel_for(mesh.nelems(), OMEGA_H_LAMBDA(int e) {
    gids(e) = e;
    ppe(e) = e == initial_parent ? 1 : 0;
  });
  auto policy = p::TeamPolicyAuto(10000, 32);
  std::unique_ptr<PS> particles(new p::SellCSigma<Particle>(policy, INT_MAX,
      1024, mesh.nelems(), 1, ppe, gids));
  auto origin = particles->get<0>();
  auto destination = particles->get<1>();
  auto ids = particles->get<2>();
  o::Write<o::LO> active_slot(1, -1);
  auto initialize = PS_LAMBDA(int, int pid, bool mask) {
    if (!mask) return;
    active_slot[0] = pid;
    ids(pid) = 20359;
    for (int d = 0; d < 3; ++d) {
      origin(pid,d) = start[d];
      destination(pid,d) = target[d];
    }
  };
  p::parallel_for(particles.get(), initialize, "restore_failing_particle");
  const int slot = o::HostRead<o::LO>(active_slot)[0];
  if (slot < 0) return 2;
  RecordWalk handler{verbose, slot, start, target, tolerance};
  o::Write<o::LO> elements, next, faces, exits, done;
  o::Write<o::Real> points;
  auto measures = measure_elements_real(&mesh);
  const bool found = p::trace_particle_through_mesh(mesh, particles.get(), origin,
      destination, ids, elements, next, true, faces, points, exits, 200, true,
      handler, measures, done, tolerance);
  const int parent = o::HostRead<o::LO>(elements)[slot];
  if (!found || parent < 0) return 1;
  const auto bcc = barycentric(vertices, coordinates, parent, target);
  const bool contains = p::all_positive(bcc, tolerance);
  std::printf("RESULT parent=%d contains_destination=%d barycentric=(%.17g, %.17g, %.17g, %.17g)\n",
              parent, contains, bcc[0], bcc[1], bcc[2], bcc[3]);
  return contains ? 0 : 1;
}
