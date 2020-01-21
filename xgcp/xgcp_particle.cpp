#include "xgcp_particle.hpp"
#include <random>
#include <chrono>

//TODO: decide how to handle this
#define PARTICLE_SEED 512*512

namespace xgcp {
  void setInitialPtclCoords(Mesh& m, PS_I* ptcls);
  void setPtclIds(PS_I* ptcls);

  PS_I* initializeIons(Mesh& m, ps::gid_t nPtcls, PS_I::kkLidView ptcls_per_elem,
                       PS_I::kkGidView element_gids) {
    ps::lid_t nElems = m.nelems();
    //TODO: Read PS parameters from some input source
    //'sigma', 'V', and the 'policy' control the layout of the PS structure
    //in memory and can be ignored until performance is being evaluated.  These
    //are reasonable initial settings for OpenMP.

    const int sigma = INT_MAX; // full sorting
    const int V = 1024;
    Kokkos::TeamPolicy<PS_I::execution_space> policy(10000, 32);
    PS_I* ptcls = new ps::SellCSigma<Ion>(policy, sigma, V, nElems, nPtcls,
                                          ptcls_per_elem, element_gids);
    setInitialPtclCoords(m, ptcls);
    setPtclIds(ptcls);
    return ptcls;
  }

  void setInitialPtclCoords(Mesh& m, PS_I* ptcls) {
    //Randomly distrubite particles within each element (uniformly within the element)
    //Create a deterministic generation of random numbers on the host with 3 number per particle
    //Use comm rank to make each process different
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    comm_rank++;
    o::HostWrite<o::Real> rand_num_per_ptcl(3*ptcls->capacity());
    std::default_random_engine generator(PARTICLE_SEED / comm_rank);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < ptcls->capacity(); ++i) {
      o::Real x = dist(generator);
      o::Real y = dist(generator);
      if (x+y > 1) {
        x = 1-x;
        y = 1-y;
      }
      rand_num_per_ptcl[3*i] = x;
      rand_num_per_ptcl[3*i+1] = y;
      rand_num_per_ptcl[3*i+2] = dist(generator);
    }

    //Torodial section bounds
    o::Real majorAngle = m.getMajorPlaneAngle();
    o::Real minorAngle = m.getMinorPlaneAngle();

    o::Write<o::Real> rand_nums(rand_num_per_ptcl);
    auto cells2nodes = m->get_adj(o::FACE, o::VERT).ab2b;
    auto nodes2coords = m->coords();
    //set particle positions and parent element ids
    auto x_ps_d = ptcls->get<PTCL_COORDS>();
    auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
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
        x_ps_d(pid,2) = (majorAngle - minorAngle)/2*(r3 + 1) + minorAngle;
        if (x_ps_d(pid,2) > majorAngle || x_ps_d(pid,2) < minorAngle)
          printf("[ERROR] Particle outside torodial section\n");
      }
    };
    ps::parallel_for(ptcls, lamb);
  }

  void setPtclIds(PS_I* ptcls) {
    auto pid_d = ptcls->get<PTCL_IDS>();
    auto setIDs = PS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
      pid_d(pid) = pid;
    };
    ps::parallel_for(ptcls, setIDs);
  }

}
