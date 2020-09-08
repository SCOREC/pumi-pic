
#include "pumipic_mesh.hpp"
#include <particle_structs.hpp>
#include "pumipic_lb.hpp"

namespace pumipic {
  /* High level particle-mesh migration & rebuild operations */

  /* Migrate/rebuild particle structure with particle load balancing
     mesh - picpart mesh
     ptcls - particle structure
     new_elems - new assignment of mesh elements for each particle
     tol - target imbalance for load balancing. (Example 5% imbalance has value 1.05)
     step_factor - (optional) The rate of diffusion for load balancer
  */
  template <class PS>
  void migrate_lb_ptcls(Mesh& mesh, PS* ptcls, Omega_h::LOs new_elems,
                  float tol, float step_factor = 0.5);

  /* Migrate/rebuild particle structure
     mesh - picpart mesh
     ptcls - particle structure
     new_elems - new assignment of mesh elements for each particle
     tol - target imbalance for load balancing. (Example 5% imbalance has value 1.05)
     step_factor - (optional) The rate of diffusion for load balancer
  */
  template <class PS>
  void migrate_ptcls(Mesh& mesh, PS* ptcls, Omega_h::LOs new_elems);


  template <class PS>
  void setUnsafeProcs(Mesh& mesh, PS* ptcls, Omega_h::LOs elems,
                      typename PS::kkLidView new_elems, typename PS::kkLidView new_procs) {
    auto owners = mesh.entOwners(mesh->dim());
    auto safe = mesh.safeTag();
    int comm_rank = mesh.comm()->rank();
    auto setUnsafePtcls = PS_LAMBDA(const int elm, const int ptcl, const bool mask) {
      new_procs[ptcl] = comm_rank;
      const int nelm = elems[ptcl];
      new_elems[ptcl] = nelm;
      if (mask) {
        if (nelm != -1) {
          const bool is_safe = safe[nelm];
          const int owner = owners[nelm];
          if (!is_safe)
            new_procs[ptcl] = owner;
        }
      }
    };
    parallel_for(ptcls, setUnsafePtcls, "setUnsafePtcls");
  }
  template <class PS>
  void migrate_lb_ptcls(Mesh& mesh, PS* ptcls, Omega_h::LOs elems,
                  float tol, float step_factor) {
    typename PS::kkLidView new_elems("ps_element_ids", ptcls->capacity());
    typename PS::kkLidView new_procs("ps_process_ids", ptcls->capacity());
    setUnsafeProcs(mesh, ptcls, elems, new_elems, new_procs);
    ParticleBalancer* balancer = mesh.ptclBalancer();
    balancer->repartition(mesh, ptcls, tol, new_elems, new_procs, step_factor);
    ptcls->migrate(new_elems, new_procs);
  }

  template <class PS>
  void migrate_ptcls(Mesh& mesh, PS* ptcls, Omega_h::LOs elems) {
    typename PS::kkLidView new_elems("ps_element_ids", ptcls->capacity());
    typename PS::kkLidView new_procs("ps_process_ids", ptcls->capacity());
    setUnsafeProcs(mesh, ptcls, elems, new_elems, new_procs);
    ptcls->migrate(new_elems, new_procs);
  }

}
