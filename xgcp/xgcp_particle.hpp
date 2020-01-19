#pragma once
#include "xgcp_types.hpp"
#include "xgcp_mesh.hpp"
namespace xgcp {
  //Get the total number of particles across all processes
  template <class PS>
  ps::gid_t getGlobalParticleCount(PS* ptcls);

  /* Create the particle structure of ions and set initial values
     m - the XGCp mesh
     nPtcls - number of particles
     ptcls_per_elem - number of particles in each mesh elements
     element_gids - global IDs of mesh elements
   */
  PS_I* initializeIons(Mesh& m, ps::gid_t nPtcls, PS_I::kkLidView ptcls_per_elem,
                       PS_I::kkGidView element_gids);
  /* Create the particle structure of electrons and set initial values
     TODO
   */
  PS_E* initializeElectrons();

  /* Performs adjacency search and migrates/rebuilds the particle structure

   */
  template <class PS>
  void search(Mesh& mesh, PS* ptcls);

  template <class PS>
  ps::gid_t getGlobalParticleCount(PS* ptcls) {
    ps::gid_t np = ptcls->nPtcls(), total_ptcls;
    MPI_Allreduce(&np, &total_ptcls, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return total_ptcls;
  }
}
