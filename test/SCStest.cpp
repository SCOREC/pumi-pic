#include <iostream>
#include <vector>
#include <fstream>
#include "Omega_h_file.hpp"
#include <Omega_h_mesh.hpp>
#include "Omega_h_for.hpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_adjacency.hpp"
#include <particle_structs.hpp>
#include <Kokkos_Core.hpp>
#include "pumipic_mesh.hpp"

//#include "pumipic_library.hpp"

namespace o = Omega_h;
namespace p = pumipic;
namespace ps = particle_structs;

using pumipic::fp_t;
using pumipic::Vector3d;
using ps::lid_t;
using ps::SellCSigma;
using ps::MemberTypes;

typedef MemberTypes < Vector3d, Vector3d, int,  Vector3d, Vector3d, 
   int, fp_t, fp_t, int, fp_t, fp_t, int, fp_t> Particle;
enum {PTCL_POS, PTCL_NEXT_POS, PTCL_ID, PTCL_VEL, PTCL_EFIELD, PTCL_CHARGE,
 PTCL_WEIGHT, PTCL_FIRST_IONIZEZ, PTCL_PREV_IONIZE, PTCL_FIRST_IONIZET, 
 PTCL_PREV_RECOMBINE, PTCL_HIT_NUM, PTCL_VMAG_NEW};

//typedef MemberTypes < Vector3d, Vector3d, int> Particle;
typedef ps::ParticleStructure<Particle> PS;

o::Mesh readMesh(char* meshFile, o::Library& lib) {
  const auto rank = lib.world()->rank();
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    if(!rank)
    std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    if(!rank)
      std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self(), true);
  } else {
    if(!rank)
      std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

void psTestRun(PS* ptcls) {
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto next_pos_ps = ptcls->get<PTCL_NEXT_POS>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto ps_charge = ptcls->get<PTCL_CHARGE>();
  auto ps_weight = ptcls->get<PTCL_WEIGHT>();
  auto ps_hitNum = ptcls->get<PTCL_HIT_NUM>();
  auto ps_newVelMag = ptcls->get<PTCL_VMAG_NEW>();

  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto weight = ps_weight(pid);
      auto vel = p::makeVector3(pid, vel_ps );
      if(ptcl < 10)
        printf("ptcl %d wt %g vel %g %g %g \n", ptcl, weight, vel[0], vel[1], vel[2]);
    }
  };
  ps::parallel_for(ptcls, lambda);
}

int main(int argc, char** argv) {
  auto start_sim = std::chrono::system_clock::now(); 
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  if(argc < 2){
    std::cout << "Usage: " << argv[0] << " <mesh>\n"; 
    exit(1); 
  }
  printf(" Mesh file %s\n", argv[1]);
  auto full_mesh = readMesh(argv[1], lib);
  Omega_h::HostWrite<Omega_h::LO> host_owners(full_mesh.nelems());
  for (int i = 0; i < full_mesh.nelems(); ++i)
    host_owners[i] = 0;
  Omega_h::Write<Omega_h::LO> owner(host_owners);
  p::Mesh picparts(full_mesh, owner);
  o::Mesh* mesh = picparts.mesh();
  printf("Mesh loaded with verts %d edges %d faces %d elements %d\n",
     mesh->nverts(), mesh->nedges(), mesh->nfaces(), mesh->nelems());

  o::Int ne = mesh->nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = mesh_element_gids[i]; 
  });
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    int num = 0;
    if(i<5) num = i;
    ptcls_per_elem(i) = num;
  });
  int numPtcls = 10; //not matching with that in elems !

  const int sigma = INT_MAX; // full sorting
  const int V = 128; //1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  PS* ptcls = new SellCSigma<Particle>(policy, sigma, V, ne, 
    numPtcls, ptcls_per_elem, element_gids);
  psTestRun(ptcls); 
  printf("Done PS test \n");

  return 0;
}
 
