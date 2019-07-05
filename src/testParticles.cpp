
#include "Omega_h_for.hpp"
#include "testMesh.hpp"
#include "testParticles.hpp"

// Initialized in only one element
void testParticles::testDefineParticles(int numPtcls, int elId) {

  o::Int ne = mesh.nelems();
  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  //Element gids is left empty since there is no partitioning of the mesh yet
  SCS::kkGidView element_gids("elem_gids", 0);
  if(elId>=0) {
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
      ptcls_per_elem(i) = 0;
      if (i == elId) {
        ptcls_per_elem(i) = numPtcls;
        printf(" Ptcls in elId %d\n", elId);
      }
    });
  }

  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
    if (np > 0)
      printf(" ptcls/elem[%d] %d\n", i, np);

  });
  //'sigma', 'V', and the 'policy' control the layout of the SCS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
    printf("Constructing Particles\n");
  //Create the particle structure
  scs = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
                   ptcls_per_elem, element_gids);
}


void testParticles::testInitImpurityPtclsInADir(o::LO numPtcls) {
  o::LO initEl = 1;
  testDefineParticles(numPtcls, initEl);
  printf("Constructed Particles\n");
}

