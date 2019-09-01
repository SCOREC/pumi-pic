#include <Omega_h_mesh.hpp>
#include "pumipic_kktypes.hpp"
#include "pumipic_adjacency.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include "pumipic_mesh.hpp"
#include <fstream>

using particle_structs::fp_t;
using particle_structs::lid_t;
using particle_structs::Vector3d;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;
using particle_structs::distribute_name;
using particle_structs::elemCoords;

namespace o = Omega_h;
namespace p = pumipic;

//To demonstrate push and adjacency search we store:
//-two fp_t[3] arrays, 'Vector3d', for the current and
// computed (pre adjacency search) positions, and
//-an integer to store the particles id
typedef MemberTypes<Vector3d, Vector3d, int> Particle;
typedef SellCSigma<Particle> SCS;

void setPtclIds(SCS* scs) {
  auto pid_d = scs->get<2>();
  auto setIDs = SCS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    if(mask)
      pid_d(pid) = pid;
  };
  scs->parallel_for(setIDs);
}

void updatePtclPositions(SCS* scs) {
  auto x_scs_d = scs->get<0>();
  auto xtgt_scs_d = scs->get<1>();
  auto updatePtclPos = SCS_LAMBDA(const int&, const int& pid, const bool&) {
    x_scs_d(pid,0) = xtgt_scs_d(pid,0);
    x_scs_d(pid,1) = xtgt_scs_d(pid,1);
    x_scs_d(pid,2) = xtgt_scs_d(pid,2);
    xtgt_scs_d(pid,0) = 0;
    xtgt_scs_d(pid,1) = 0;
    xtgt_scs_d(pid,2) = 0;
  };
  scs->parallel_for(updatePtclPos);
}

void rebuild(p::Mesh& picparts, SCS* scs, o::LOs elem_ids, const bool output) {
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto ids = scs->get<2>();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(output && mask > 0)
      printf("elem_ids[%d] %d ptcl_id:%d\n", pid, elem_ids[pid], ids(pid));
  };
  scs->parallel_for(printElmIds);

  SCS::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);
  SCS::kkLidView scs_process_ids("scs_process_ids", scs_capacity);
  Omega_h::LOs is_safe = picparts.safeTag();
  Omega_h::LOs elm_owners = picparts.entOwners(picparts.dim());
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      int new_elem = elem_ids[pid];
      scs_elem_ids(pid) = new_elem;
      scs_process_ids(pid) = comm_rank;
      if (new_elem != -1 && is_safe[new_elem] == 0) {
        scs_process_ids(pid) = elm_owners[new_elem];
      }
    }
  };
  scs->parallel_for(lamb);

  scs->migrate(scs_elem_ids, scs_process_ids);

  printf("SCS on rank %d has Elements: %d. Ptcls %d. Capacity %d. Rows %d.\n"
         , comm_rank, scs->nElems(), scs->nPtcls(), scs->capacity(), scs->numRows());
  ids = scs->get<2>();
  if (output) {
    auto printElms = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0)
        printf("Rank %d Ptcl: %d has Element %d and id %d\n", comm_rank, pid, e, ids(pid));
    };
    scs->parallel_for(printElms);
  }
}

void search(p::Mesh& picparts, SCS* scs, bool output=false) {
  o::Mesh* mesh = picparts.mesh();
  assert(scs->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  Kokkos::Timer timer;
  auto x = scs->get<0>();
  auto xtgt = scs->get<1>();
  auto pid = scs->get<2>();
  o::Write<o::Real> xpoints_d(3 * scsCapacity, "intersection points");
  o::Write<o::LO> xface_id(scsCapacity, "intersection faces");
  bool isFound = p::search_mesh_2d<Particle>(*mesh, scs, x, xtgt, pid, elem_ids,
                                          xpoints_d, maxLoops);
  fprintf(stderr, "search_mesh (seconds) %f\n", timer.seconds());
  assert(isFound);
  //rebuild the SCS to set the new element-to-particle lists
  timer.reset();
  rebuild(picparts, scs, elem_ids, output);
  fprintf(stderr, "rebuild (seconds) %f\n", timer.seconds());
}

o::Mesh readMesh(const char* meshFile, o::Library& lib) {
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self());
  } else {
    std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

void particleSearch(p::Mesh& picparts,
    const int parentElm, const double* start, const double* end) {
  o::Mesh* mesh = picparts.mesh();
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());

  /* Particle data */
  const auto ne = mesh->nelems();
  const auto numPtcls = 1;
  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  SCS::kkGidView element_gids("element_gids", ne);
  //place one particle in element 0
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = mesh_element_gids[i];
    ptcls_per_elem(i) = (i == parentElm);
    printf("ppe[%d] %d\n", i, ptcls_per_elem(i));
  });

  const int sigma = INT_MAX; // full sorting
  const int V = 32;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  //Create the particle structure
  SellCSigma<Particle>* scs = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
                                                       ptcls_per_elem, element_gids);
  auto cells2nodes = mesh->get_adj(o::FACE, o::VERT).ab2b;
  auto nodes2coords = mesh->coords();
  //set particle positions
  auto x_scs_d = scs->get<0>();
  auto x_scs_tgt = scs->get<1>();
  const double ptclStart[2] = {start[0],start[1]};
  const double ptclEnd[2] = {end[0],end[1]};
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask && !pid) {
      for(int i=0; i<2; i++) {
        x_scs_d(pid,i) = ptclStart[i];
        x_scs_tgt(pid,i) = ptclEnd[i];
      }
      printf("pid %d elm %d src %f %f dest %f %f\n",
          pid, e, x_scs_d(pid,0), x_scs_d(pid,1),
          x_scs_tgt(pid,0), x_scs_tgt(pid,1));
    }
  };
  scs->parallel_for(lamb);
  setPtclIds(scs);
  search(picparts,scs);
  auto printPtclElm = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask) {
      printf("pid %d elm %d (x,y) %f %f\n",
          pid, e, x_scs_d(pid, 0), x_scs_d(pid, 1));
    }
  };
  scs->parallel_for(printPtclElm);

  delete scs;
}

int comm_rank, comm_size;

void testTri8(Omega_h::Library& lib, std::string meshDir) {
  const auto meshName = meshDir+"/plate/tri8_parDiag.osh";
  auto full_mesh = readMesh(meshName.c_str(), lib);
  Omega_h::HostWrite<Omega_h::LO> host_owners(full_mesh.nelems());
  for (int i = 0; i < full_mesh.nelems(); ++i)
    host_owners[i] = 0;
  Omega_h::Write<Omega_h::LO> owner(host_owners);
  pumipic::Input input(full_mesh, pumipic::Input::PARTITION, owner, pumipic::Input::FULL,
                       pumipic::Input::BFS);
  p::Mesh picparts(input);
  for (int i = 0; i <= full_mesh.dim(); ++i)
    assert(picparts.nents(i) == full_mesh.nents(i));

  //Create Picparts with the full mesh
  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts(); //caching adjacency info

  if (comm_rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh->nverts(), mesh->nedges(),
           mesh->nfaces(), mesh->nelems());

  { // start at a vertex and go along an adjacent edge
    const auto parentElm = 0;
    const double start[2] = {.5,.5};
    const double end[2]  = {.8,.8};
    particleSearch(picparts,parentElm,start,end);
  }
  printf("\n\n");
  { // start at a vertex and go through a bounded triangle
    const auto parentElm = 0;
    const double start[2] = {.5,.5};
    const double end[2]  = {.8,0};
    particleSearch(picparts,parentElm,start,end);
  }
  printf("\n\n");
  { // start and stop along the same edge
    const auto parentElm = 0;
    const double start[2] = {.25,.25};
    const double end[2]  = {.4,.4};
    particleSearch(picparts,parentElm,start,end);
  }
}

void testItg24k(Omega_h::Library& lib, std::string meshDir) {
  const auto meshName = meshDir+"/xgc/itg24k.osh";
  auto full_mesh = readMesh(meshName.c_str(), lib);
  Omega_h::HostWrite<Omega_h::LO> host_owners(full_mesh.nelems());
  for (int i = 0; i < full_mesh.nelems(); ++i)
    host_owners[i] = 0;
  Omega_h::Write<Omega_h::LO> owner(host_owners);
  pumipic::Input input(full_mesh, pumipic::Input::PARTITION, owner, pumipic::Input::FULL,
                       pumipic::Input::BFS);
  p::Mesh picparts(input);
  for (int i = 0; i <= full_mesh.dim(); ++i)
    assert(picparts.nents(i) == full_mesh.nents(i));

  //Create Picparts with the full mesh
  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts(); //caching adjacency info

  if (comm_rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh->nverts(), mesh->nedges(),
           mesh->nfaces(), mesh->nelems());

  {
    const auto parentElm = 2554;
    const double start[2] = {1.514768,0.261045};
    const double end[2]  = {1.512270,0.257226};
    particleSearch(picparts,parentElm,start,end);
  }
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if( argc != 2 ) {
    std::cout << "Usage: " << argv[0] << " <testMeshDir>\n";
    exit(1);
  }
  std::string meshDir(argv[1]);
  testTri8(lib,meshDir);
  testItg24k(lib,meshDir);
  if (!comm_rank)
    fprintf(stderr, "done\n");
  return 0;
}
