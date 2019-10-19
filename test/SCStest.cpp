#include <iostream>
#include <vector>
#include <fstream>
#include "Omega_h_file.hpp"
#include <Omega_h_mesh.hpp>
#include "pumipic_kktypes.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include "pumipic_library.hpp"
#include "pumipic_mesh.hpp"

using particle_structs::fp_t;
using particle_structs::lid_t;
using particle_structs::Vector3d;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;

namespace o = Omega_h;
namespace p = pumipic;
typedef MemberTypes < Vector3d, Vector3d, int> Particle;
typedef SellCSigma<Particle> SCS;
   
int main(int argc, char** argv) {
  auto start_sim = std::chrono::system_clock::now(); 
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  if(argc < 1){
    std::cout << "Usage: " << argv[0] << " <mesh>\n"; 
    exit(1); 
  }

  auto full_mesh = readMesh(argv[1], lib);
  Omega_h::HostWrite<Omega_h::LO> host_owners(full_mesh.nelems());
  for (int i = 0; i < full_mesh.nelems(); ++i)
    host_owners[i] = 0;
  Omega_h::Write<Omega_h::LO> owner(host_owners);
  p::Mesh picparts(full_mesh, owner);
  o::Mesh* mesh = picparts.mesh();
  printf("Mesh loaded with verts %d edges %d faces %d elements %d\n",
     mesh->nverts(), mesh->nedges(), mesh->nfaces(), mesh->nelems());

  int numPtcls = 10;
  o::Int ne = mesh->nelems();
  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  SCS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = mesh_element_gids[i]; 
  });
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    int num = 0;
    if(i<5) num = i;
    ptcls_per_elem(i) = num;
  });
  const int sigma = INT_MAX; // full sorting
  const int V = 128; //1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  SellCSigma<Particle>* scs = new SellCSigma<Particle>(policy, sigma, V, ne, 
    numPtcls, ptcls_per_elem, element_gids);
  const auto scsCapacity = scs->capacity();
  printf("Done SCS creation \n");
  return 0;
}
 
