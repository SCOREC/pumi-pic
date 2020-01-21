#pragma once
#include "xgcp_types.hpp"
#include "xgcp_mesh.hpp"
#include <pumipic_adjacency.hpp>
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
  template <typename PS>
  void search(Mesh& mesh, PS_I* ptcls);

  /* Migrate particles and rebuild particle structure

   */
  template <typename PS>
  void rebuild(Mesh& mesh, PS* ptcls, o::LOs elem_ids);

  template <class PS>
  ps::gid_t getGlobalParticleCount(PS* ptcls) {
    ps::gid_t np = ptcls->nPtcls(), total_ptcls;
    MPI_Allreduce(&np, &total_ptcls, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return total_ptcls;
  }

  template <typename PS>
  void search(Mesh& mesh, PS* ptcls) {
    Omega_h::LO maxLoops = 200;
    const auto psCapacity = ptcls->capacity();
    o::Write<o::LO> elem_ids(psCapacity, -1);
    auto x_ps_d = ptcls->template get<0>();
    auto xtgt_ps_d = ptcls->template get<1>();
    auto pid = ptcls->template get<2>();
    bool isFound = p::search_mesh_2d(*(mesh.omegaMesh()), ptcls, x_ps_d, xtgt_ps_d,
                                     pid, elem_ids, maxLoops);
    assert(isFound);
    rebuild(mesh, ptcls, o::LOs(elem_ids));
  }

  template <typename PS>
  void updatePtclPositions(PS* ptcls) {
    auto x_ps_d = ptcls->template get<0>();
    auto xtgt_ps_d = ptcls->template get<1>();
    auto updatePtclPos = PS_LAMBDA(const int&, const int& pid, const bool&) {
      x_ps_d(pid,0) = xtgt_ps_d(pid,0);
      x_ps_d(pid,1) = xtgt_ps_d(pid,1);
      x_ps_d(pid,2) = xtgt_ps_d(pid,2);
      xtgt_ps_d(pid,0) = 0;
      xtgt_ps_d(pid,1) = 0;
      xtgt_ps_d(pid,2) = 0;
    };
    ps::parallel_for(ptcls, updatePtclPos);
  }

  template <typename PS>
  void rebuild(Mesh& mesh, PS* ptcls, o::LOs elem_ids) {
    p::Mesh* picparts = mesh.pumipicMesh();
    updatePtclPositions(ptcls);
    const int ps_capacity = ptcls->capacity();
    auto xs = ptcls->template get<0>();
    //Gather new element and new process for migrate/rebuild
    PS_I::kkLidView ps_elem_ids("ps_elem_ids", ps_capacity);
    PS_I::kkLidView ps_process_ids("ps_process_ids", ps_capacity);
    Omega_h::LOs is_safe = picparts->safeTag();
    Omega_h::LOs elm_owners = picparts->entOwners(picparts->dim());
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    int mr_local = mesh.meshRank();
    int gr_local = mesh.groupRank();
    int tr_local = mesh.torodialRank();
    int ms_local = mesh.meshSize();
    int gs_local = mesh.groupSize();
    int ts_local = mesh.torodialSize();
    int nplanes = mesh.nplanes();
    auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask) {
        //Gather element
        int new_elem = elem_ids[pid];
        ps_elem_ids(pid) = new_elem;
        //Calculate new process by each process dimension
        const int mesh_rank = new_elem != -1 ? mr_local : elm_owners[new_elem];
        const int torodial_rank = xs(pid,2) * nplanes / (2 * M_PI);
        const int group_rank = gr_local;
        ps_process_ids(pid) = getWorldRank(torodial_rank, mesh_rank, group_rank,
                                           ts_local, ms_local, gs_local);
      }
    };
    ps::parallel_for(ptcls, lamb);

    ptcls->migrate(ps_elem_ids, ps_process_ids);

    //Check to see if particles are all in correct places
    fp_t major_phi = mesh.getMajorPlaneAngle();
    fp_t minor_phi = mesh.getMinorPlaneAngle();
    auto coords = ptcls->template get<0>();
    auto pids = ptcls->template get<2>();
    auto checkPtcls = PS_LAMBDA(const int& e, const int& p, const bool& m) {
      if (m) {
        auto pid = pids(p);
        if (!is_safe[e])
          printf("Particle %d is in an unsafe element\n", pid);
        if (coords(p,2) < minor_phi || coords(p,2) > major_phi)
          printf("Particle %d is outside torodial section [%f < %f < %f]\n", pid,
                 minor_phi, coords(p,2), major_phi);
      }
    };
    ps::parallel_for(ptcls, checkPtcls, "check particles");
  }
}
