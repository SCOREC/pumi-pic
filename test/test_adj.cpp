#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_bbox.hpp>
#include "pumipic_adjacency.hpp"
#include <random>

#define ELEMENT_SEED 1024*1024
#define PARTICLE_SEED 512*512

namespace o = Omega_h;
namespace p = pumipic;

using p::fp_t;
using p::Vector3d;
/* Define particle types
   0 = current position
   1 = pushed position
   2 = ids
   3 = direction of motion
 */
typedef p::MemberTypes<Vector3d, Vector3d, int, Vector3d> Particle;
typedef p::ParticleStructure<Particle> PS;

int setSourceElements(o::Mesh mesh, PS::kkLidView ppe, const int numPtcls) {
  auto numPpe = numPtcls / mesh.nelems();
  auto numPpeR = numPtcls % mesh.nelems();
  auto cells2nodes = mesh.ask_down(mesh.dim(), o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  o::LO lastElm = mesh.nelems() - 1;
  o::parallel_for(mesh.nelems(), OMEGA_H_LAMBDA(const int i) {
      ppe[i] = numPpe + ( (i==lastElm) * numPpeR );
    });
  Omega_h::LO totPtcls = 0;
  Kokkos::parallel_reduce(ppe.size(), OMEGA_H_LAMBDA(const int i, Omega_h::LO& lsum) {
      lsum += ppe[i];
    }, totPtcls);
  assert(totPtcls == numPtcls);
  return totPtcls;
}


PS* create_particle_structure(o::Mesh mesh, p::lid_t numPtcls) {
  Omega_h::Int ne = mesh.nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = i;
  });
  int actualParticles = setSourceElements(mesh, ptcls_per_elem, numPtcls);
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
  });

  //'sigma', 'V', and the 'policy' control the layout of the PS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  //Create the particle structure
  return new p::SellCSigma<Particle>(policy, sigma, V, ne, actualParticles,
                                     ptcls_per_elem, element_gids);
}
void init2DInternal(o::Mesh mesh, PS* ptcls) {
  //Randomly distrubite particles within each element (uniformly within the element)
  //Create a deterministic generation of random numbers on the host with 2 number per particle

  o::HostWrite<o::Real> rand_num_per_ptcl(3*ptcls->capacity());
  std::default_random_engine generator(PARTICLE_SEED);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < ptcls->capacity(); ++i) {
    o::Real x = dist(generator);
    o::Real y = dist(generator);
    o::Real ang = dist(generator);
    if (x+y > 1) {
      x = 1-x;
      y = 1-y;
    }
    rand_num_per_ptcl[3*i] = x;
    rand_num_per_ptcl[3*i+1] = y;
    rand_num_per_ptcl[3*i+2] = ang * 2* M_PI;
  }
  o::Write<o::Real> rand_nums(rand_num_per_ptcl);
  auto cells2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  //set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>();
  auto pids = ptcls->get<2>();
  auto motion = ptcls->get<3>();
  o::Reals elmArea = measure_elements_real(&mesh);
  
  auto lamb = PS_LAMBDA(const int e, const int pid, const int mask) {
    if(mask > 0) {
      auto elmVerts = o::gather_verts<3>(cells2nodes, o::LO(e));
      auto vtxCoords = o::gather_vectors<3,2>(nodes2coords, elmVerts);
      o::Real r1 = rand_nums[3*pid];
      o::Real r2 = rand_nums[3*pid+1];
      o::Real r3 = rand_nums[3*pid+2];
      // X = A + r1(B-A) + r2(C-A)
      for (int i = 0; i < 2; i++)
        x_ps_d(pid,i) = vtxCoords[0][i] + r1 * (vtxCoords[1][i] - vtxCoords[0][i])
                                        + r2 * (vtxCoords[2][i] - vtxCoords[0][i]);
      x_ps_d(pid,2) = 0;

      auto ptclOrigin = p::makeVector2(pid, x_ps_d);
      Omega_h::Vector<3> faceBcc;
      p::barycentric_tri(elmArea[e], vtxCoords, ptclOrigin, faceBcc);
      assert(p::all_positive(faceBcc));
      if (!p::all_positive(faceBcc)) printf("FAILURE\n");
      motion(pid, 0) = cos(r3);
      motion(pid, 1) = sin(r3);
      motion(pid, 2) = 0;
      pids(pid) = pid;
    }
  };
  ps::parallel_for(ptcls, lamb);
}

void init3DInternal(o::Mesh mesh, PS* ptcls) {
  //Randomly distrubite particles within each element (uniformly within the element)
  //Method taken from http://vcg.isti.cnr.it/jgt/tetra.htm
  //Create a deterministic generation of random numbers on the host with 2 number per particle

  o::HostWrite<o::Real> rand_num_per_ptcl(5*ptcls->capacity());
  std::default_random_engine generator(PARTICLE_SEED);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < ptcls->capacity(); ++i) {
    o::Real x = dist(generator);
    o::Real y = dist(generator);
    o::Real z = dist(generator);
    o::Real ang = dist(generator);
    o::Real r = dist(generator);
    if (x+y > 1) {
      x = 1-x;
      y = 1-y;
    }
    if (y+z > 1) {
      o::Real tmp = z;
      z = 1 - x - y;
      y = 1 - tmp;
    }
    else if (x + y + z > 1) {
      o::Real tmp = z;
      z = x + y + z  - 1;
      x = 1 - y - tmp;
    }
    rand_num_per_ptcl[5*i] = x;
    rand_num_per_ptcl[5*i+1] = y;
    rand_num_per_ptcl[5*i+2] = z;
    rand_num_per_ptcl[5*i+3] = ang * 2 * M_PI;
    rand_num_per_ptcl[5*i+4] = r * 2 - 1;
  }
  o::Write<o::Real> rand_nums(rand_num_per_ptcl);
  auto cells2nodes = mesh.ask_down(o::REGION, o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  //set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>(); 
  auto x_ps_tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  auto motion = ptcls->get<3>();
  
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = o::gather_verts<4>(cells2nodes, o::LO(e));
      auto vtxCoords = o::gather_vectors<4,3>(nodes2coords, elmVerts);
      const o::Real s = rand_nums[5*pid];
      const o::Real t = rand_nums[5*pid+1];
      const o::Real u = rand_nums[5*pid+2];
      const o::Real a = 1 - s - t - u;
      for (int i = 0; i < 3; i++) {
        x_ps_d(pid, i) = a * vtxCoords[0][i] + s * vtxCoords[1][i]
          + t * vtxCoords[2][i] + u * vtxCoords[3][i];
        x_ps_tgt(pid, i) = x_ps_d(pid, i);
      }
      const o::Real theta = rand_nums[5*pid + 3];
      const o::Real z = rand_nums[5*pid + 4];
      motion(pid, 0) = sqrt(1 - z*z) * cos(theta);
      motion(pid, 1) = sqrt(1 - z*z) * sin(theta);
      motion(pid, 2) = z;
      pids(pid) = pid;
    }
  };
  ps::parallel_for(ptcls, lamb);

}

template <int N>
o::Real determineDistance(o::Mesh mesh) {
  auto bb = o::get_bounding_box<N>(&mesh);
  char buffer[1024];
  char name[3] = {'x', 'y', 'z'};
  char* ptr = buffer + sprintf(buffer, "  Bounding Box:");
  for (int i = 0; i < N; ++i)
    ptr += sprintf(ptr, " |%c: [%f:%f]|", name[i], bb.min[i], bb.max[i]);
  fprintf(stderr, "%s\n", buffer);
  double maxDimLen = 0;
  for (int i = 0; i < N; ++i) {
    o::Real len = bb.max[i] - bb.min[i];
    if (len > maxDimLen)
      maxDimLen = len;
  }
  return maxDimLen;
}

//Initialize particle position and directions inside each element
o::Real init_internal(o::Mesh mesh, PS* ptcls) {
  if (mesh.dim() == 2) {
    init2DInternal(mesh, ptcls);
    return determineDistance<2>(mesh) / 20;
  }
  init3DInternal(mesh, ptcls);
  return determineDistance<3>(mesh) / 20;
}

//Push particles
void push_ptcls(PS* ptcls, o::Real distance) {
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto angle = ptcls->get<3>();
  auto push = PS_LAMBDA(const p::lid_t elm, const p::lid_t ptcl, const bool mask) {
    if (mask) {
      for (int i = 0; i < 3; ++i) {
        tgt(ptcl, i) = tgt(ptcl,i) + distance * angle(ptcl, i);
      }
    }
  };
  p::parallel_for(ptcls, push, "push");
}

//Tests the particle is within its parent element and then sets the particle coordinates to the target position
bool test_parent_elements(o::Mesh mesh, PS* ptcls, o::Write<o::LO> elem_ids) {
  //Use pumipic routine in search to check element
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  o::Reals elmArea = measure_elements_real(&mesh);
  o::Write<o::LO> ptcl_done(ptcls->capacity(), 0, "ptcl_done");
  auto setDone = PS_LAMBDA(const o::LO, const o::LO ptcl, const bool mask) {
    ptcl_done[ptcl] = (!mask | (elem_ids[ptcl] == -1));
  };
  p::parallel_for(ptcls, setDone, "setDone");

  o::LO failures = p::check_initial_parents(mesh, ptcls, tgt, pids, elem_ids, ptcl_done, elmArea, true);

  //Reset coordinates
  auto resetCoordinates = PS_LAMBDA(const o::LO elm, const o::LO ptcl, bool) {
    for (int i = 0; i < 3; ++i) {
      cur(ptcl, i) = tgt(ptcl, i);
    }
  };
  p::parallel_for(ptcls, resetCoordinates, "resetCoordinates");
  return failures;
}

//Checks particles that did not intersect the wall are within the mesh still
//Assumes the plate or cube mesh are used
template <int DIM>
bool check_inside_bbox(o::Mesh mesh, PS* ptcls, o::Write<o::LO> xFaces) {
  auto box = o::get_bounding_box<DIM>(&mesh);
  auto x_ps_orig = ptcls->get<0>();
  o::Write<o::LO> failures(1,0);
  auto checkInteriors = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    const o::LO face = xFaces[ptcl];
    const o::Vector<3> ptcl_pos = makeVector3(ptcl, x_ps_orig);
    if (mask && face == -1) {
      bool fail = false;
      for (int i = 0; i < DIM; ++i) {
        fail |= (ptcl_pos[i] < box.min[i]);
        fail |= (ptcl_pos[i] > box.max[i]);
      }
      Kokkos::atomic_add(&(failures[0]), (o::LO)fail);
    }
  };
  p::parallel_for(ptcls, checkInteriors, "checkInteriors");
  o::LO fails = o::HostWrite<o::LO>(failures)[0];
  if (fails != 0) {
    fprintf(stderr, "%d particles left the mesh without a model intersection calculated\n", fails);
  }
  return fails;
}

//Checks if the point is on an edge
OMEGA_H_INLINE bool check_point_on_edge(const o::Few<o::Vector<2>, 2> verts, const o::Vector<2> point, const o::Real tol, o::LO ptcl) {
  const o::Vector<2> path = point - verts[0];
  const o::Vector<2> edge = verts[1] - verts[0];
  const o::Real cross = o::cross(path, edge);

  return fabs(cross) <= tol &&
    point[0] >= min(verts[0][0], verts[1][0]) - tol &&
    point[1] >= min(verts[0][1], verts[1][1]) - tol &&
    point[0] <= max(verts[0][0], verts[1][0]) + tol &&
    point[1] <= max(verts[0][1], verts[1][1]) + tol;
}

//3D test of intersection point within the intersection face
bool check_intersections_3d(o::Mesh mesh, PS* ptcls, o::Write<o::LO> intersection_faces, o::Write<o::Real> x_points) {
  o::Write<o::LO> failures(1,0);
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto coords = mesh.coords();
  auto checkIntersections3d = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    const o::LO face = intersection_faces[ptcl];
    if (mask && face != -1) {
      //check intersection point is in the intersection face
      const auto fv2v = o::gather_verts<3>(face_verts, face);
      const auto abc = o::gather_vectors<3, 3>(coords, fv2v);
      Omega_h::Vector<3> xpoint = o::zero_vector<3>();
      for (int i =0; i < 3; ++i)
        xpoint[i] = x_points[3 * ptcl + i];
      Omega_h::Vector<3> bcc;
      bool pass = p::find_barycentric_tri_simple(abc, xpoint, bcc);
      if(pass && !p::all_positive(bcc)) {
        Kokkos::atomic_add(&(failures[0]),1);
        printf("[ERROR] Particle intersected with model boundary outside the intersection face!\n");
      }
    }
  };
  p::parallel_for(ptcls, checkIntersections3d, "checkIntersections3d");
  bool fail = o::HostWrite<o::LO>(failures)[0];
  return fail;
}

//2D test of intersection point within the intersection edge
bool check_intersections_2d(o::Mesh mesh, PS* ptcls, o::Write<o::LO> intersection_faces, o::Write<o::Real> x_points) {
  o::Write<o::LO> failures(1,0);
  const auto edge_verts =  mesh.ask_verts_of(1);
  const auto coords = mesh.coords();
  auto checkIntersections2d = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    const o::LO edge = intersection_faces[ptcl];
    if (mask && edge != -1) {
      //check intersection point is in the intersection face
      const auto ev2v = o::gather_verts<2>(edge_verts, edge);
      const auto abc = o::gather_vectors<2, 2>(coords, ev2v);
      Omega_h::Vector<2> xpoint = o::zero_vector<2>();
      for (int i =0; i < 2; ++i)
        xpoint[i] = x_points[2 * ptcl + i];
      if (!check_point_on_edge(abc, xpoint, 1e-8, ptcl)) {
        Kokkos::atomic_add(&(failures[0]),1);
        printf("[ERROR] Particle intersected with model boundary outside the intersection edge!\n");
      }
    }
  };
  p::parallel_for(ptcls, checkIntersections2d, "checkIntersections2d");
  bool fail = o::HostWrite<o::LO>(failures)[0];
  return fail; 
}

//Tests particles that hit the wall if the intersection points are correct/valid
//Also calls check_inside_bbox to check particles that did not find a wall intersection to ensure they are in the mesh
template <int DIM>
bool test_wall_intersections(o::Mesh mesh, PS* ptcls, o::Write<o::LO> elem_ids, o::Write<o::LO> intersection_faces,
                             o::Write<o::Real> x_points) {
  bool fail = false;
  o::Write<o::LO> failures(1,0);
  auto bb = o::get_bounding_box<DIM>(&mesh);
  int nvpe = DIM + 1;
  const auto elm_down = mesh.ask_down(mesh.dim(), mesh.dim() - 1);
  auto motion = ptcls->get<3>();
  //Test intersection points on face
  if (DIM == 3)
    fail |= check_intersections_3d(mesh, ptcls, intersection_faces, x_points);
  else if (DIM == 2)
    fail |= check_intersections_2d(mesh, ptcls, intersection_faces, x_points);

  //Test intersection points against motion and intersection face on parent element
  auto checkIntersections = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    const o::LO face = intersection_faces[ptcl];
    const o::LO searchElm = elem_ids[ptcl];
    if (mask && face != -1) {
      //Check the intersection point is in the direction of motion
        Omega_h::Vector<DIM> xpoint = o::zero_vector<DIM>();
        for (int i =0; i < DIM; ++i)
          xpoint[i] = x_points[DIM * ptcl + i];
      for (int i = 0; i < DIM; ++i) {
        if (fabs(xpoint[i] - bb.min[i]) < 1e-8 && motion(ptcl, i) > 0) {
          Kokkos::atomic_add(&(failures[0]),1);
          printf("[ERROR] Intersection with minimum model boundary is not in the direction of particle motion\n");

        }
        if (fabs(xpoint[i]- bb.max[i]) < 1e-8 && motion(ptcl, i) < 0) {
          Kokkos::atomic_add(&(failures[0]),1);
          printf("[ERROR] Intersection with max model boundary is not in the direction of particle motion\n");
        }
      }

      //check new parent element has the intersection face
      bool hasFace = false;
      for (int i =0; i < nvpe; ++i) {
        hasFace |= (elm_down.ab2b[nvpe*searchElm + i] == face);
      }
      if (!hasFace) {
        Kokkos::atomic_add(&(failures[0]),1);
        printf("[ERROR] Intersection face is not adjacent to parent element!\n");
      }
    }
  };
  p::parallel_for(ptcls, checkIntersections, "checkIntersections");

  fail |= o::HostWrite<o::LO>(failures)[0];
  //check non intersected particles are still in the domain
  fail |= check_inside_bbox<DIM>(mesh, ptcls, intersection_faces);
  return fail;
}

bool testInternalBCCSearch(o::Mesh mesh, PS* ptcls) {
  fprintf(stderr, "\nBeginning BCC Search test with internal particle positions\n");
  bool fail = false;
  //Initialize particle information
  o::Real distance = init_internal(mesh, ptcls);

  //Push particles
  fprintf(stderr, "  First push\n");
  push_ptcls(ptcls, distance);

  //Search
  fprintf(stderr, "  Perform search\n");
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  o::Write<o::LO> elem_ids;
  o::Write<o::LO> xFaces;
  o::Write<o::Real> xPoints;
  fail |= !p::search_mesh(mesh, ptcls, cur, tgt, pids, elem_ids, false, xFaces, xPoints, 100);

  //Test search worked
  fprintf(stderr, "  Testing resulting parent elements\n");
  fail |= test_parent_elements(mesh, ptcls, elem_ids);


  //Push particles again
  fprintf(stderr, "  Push particles again\n");
  push_ptcls(ptcls, distance);

  //Search again
  fprintf(stderr, "  Find second new parent elements\n");
  fail |= !p::search_mesh(mesh, ptcls, cur, tgt, pids, elem_ids, false, xFaces, xPoints, 100);

  //Test search again
  fprintf(stderr, "  Testing resulting parent elements again\n");
  fail |= test_parent_elements(mesh, ptcls, elem_ids);
  return fail;
}

void reflect_intersections(PS* ptcls, o::Write<o::LO> xFaces, o::Write<o::Real> xPoints) {
  auto tgt = ptcls->get<1>();
  auto dir = ptcls->get<3>();
  int dim = xPoints.size() / ptcls->capacity();
  auto reflectParticles = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    o::LO face = xFaces[ptcl];
    if (mask && face != -1) {
      for (int i = 0; i < dim; ++i) {
        tgt(ptcl, i) =  xPoints[ptcl*dim + i];
        dir(ptcl, i) *= -1;
      }
    }
  };
  p::parallel_for(ptcls, reflectParticles, "reflectParticles");
}

void delete_intersections(PS* ptcls, o::Write<o::LO> elem_ids, o::Write<o::LO> xFaces) {
  auto deleteIntersections = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    o::LO face = xFaces[ptcl];
    if (mask && face != -1) {
      elem_ids[ptcl] = -1;
    }
  };
  p::parallel_for(ptcls, deleteIntersections, "deleteIntersections");

}

bool testInternalIntersectionSearch(o::Mesh mesh, PS* ptcls) {
  fprintf(stderr, "\nBeginning BCC Search with intersection points test with internal particle positions\n");
  bool fail = false;
  //Initialize particle information
  o::Real distance = init_internal(mesh, ptcls);

  //Push particles
  fprintf(stderr, "  First push\n");
  for (int i = 0; i < 10; ++i)
    push_ptcls(ptcls, distance);

  //Search
  fprintf(stderr, "  Perform search\n");
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  o::Write<o::LO> elem_ids;
  o::Write<o::LO> xFaces;
  o::Write<o::Real> xPoints;
  fail |= !p::search_mesh(mesh, ptcls, cur, tgt, pids, elem_ids, 
                         true, xFaces, xPoints, 100);

  //Test wall intersections
  fprintf(stderr, "  Testing wall intersections\n");
  if (mesh.dim() == 2)
    fail |= test_wall_intersections<2>(mesh, ptcls, elem_ids, xFaces, xPoints);
  else
    fail |= test_wall_intersections<3>(mesh, ptcls, elem_ids, xFaces, xPoints);

  //Deal with intersections by reflection
  reflect_intersections(ptcls, xFaces, xPoints);
  
  //Test search worked
  fprintf(stderr, "  Testing resulting parent elements\n");
  fail |= test_parent_elements(mesh, ptcls, elem_ids);

  //Push particles again
  fprintf(stderr, "  Push particles again\n");
  push_ptcls(ptcls, distance);

  return fail;
  //Search again
  fprintf(stderr, "  Find second new parent elements\n");
  fail |= !p::search_mesh(mesh, ptcls, cur, tgt, pids, elem_ids, 
                          true, xFaces, xPoints, 100);

  //Test wall intersections
  fprintf(stderr, "  Testing wall intersections\n");
  if (mesh.dim() == 2)
    fail |= test_wall_intersections<2>(mesh, ptcls, elem_ids, xFaces, xPoints);
  else
    fail |= test_wall_intersections<3>(mesh, ptcls, elem_ids, xFaces, xPoints);

  //Deal with intersections by deletion
  delete_intersections(ptcls, elem_ids, xFaces);

  //Test search again
  fprintf(stderr, "  Testing resulting parent elements again\n");
  fail |= test_parent_elements(mesh, ptcls, elem_ids);

  return fail;
}

int test_search(o::Mesh mesh, p::lid_t np) {
  
  fprintf(stderr, "\n%s\n", std::string(20, '-').c_str());
  fprintf(stderr, "Begin testing with %d particles\n", np);
  
  //Create particle structure
  PS* ptcls = create_particle_structure(mesh, np);

  //Run tests
  int fails = 0;
  fails += testInternalBCCSearch(mesh, ptcls);
  fails += testInternalIntersectionSearch(mesh, ptcls);
  // fails += testEdgeBCCSearch(mesh, ptcls);
  // fails += testEdgeIntersectionSearch(mesh, ptcls);

  delete ptcls;
  return fails;
}

int main(int argc, char** argv) {

  if(argc != 2)
  {
    fprintf(stderr, "Usage: %s <mesh>\n", argv[0]);
    return 1;
  }

  //Initialize Omega_h and read in mesh
  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.world());

  int fails = 0;
  fails += test_search(mesh, 100);
  fails += test_search(mesh, 1000000);

  if (fails == 0) {
    fprintf(stderr, "\nAll Tests Passed\n");
  }
  return fails;
}
