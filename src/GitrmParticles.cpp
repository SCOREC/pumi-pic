
#include "pumipic_adjacency.hpp"
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "GitrmMesh.hpp"
#include "unit_tests.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <thread>


#define NUM_ITERATIONS 20

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



typedef MemberTypes<Vector3d, Vector3d, int > Particle;  //FIXME


//TODO remove mesh argument, once Singleton gm is used
GitrmParticles::GitrmParticles(o::Mesh &m):
  mesh(m) {
  defineParticles();
}

GitrmParticles::~GitrmParticles(){
  delete scs;
}


void GitrmParticles::defineParticles(){

  //GitrmMesh &gm = GitrmMesh::getInstance();
  //o::Mesh &mesh = gm.mesh;
  auto ne = mesh.nelems();

 /* Particle data */
  const int numPtcls = ne;  //TODO

  fprintf(stderr, "number of elements %d number of particles %d\n",
      ne, numPtcls);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  for(int i=0; i<ne; i++)
    ptcls_per_elem[i] = 1;
  for(int i=0; i<numPtcls; i++)
    ids[i].push_back(i);

  //'sigma', 'V', and the 'policy' control the layout of the SCS structure 
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  const bool debug = false;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 4);
  fprintf(stderr, "Sell-C-sigma C %d V %d sigma %d\n", policy.team_size(), V, sigma);
  //Create the particle structure
  scs = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
                   ptcls_per_elem,
                   ids, debug);
  delete [] ptcls_per_elem;
  delete [] ids;
  //Set initial and target positions so search will
  // find the parent elements
  setInitialPtclCoords(mesh, scs);
  setPtclIds(scs);
}

