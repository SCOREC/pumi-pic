#pragma once
#include <particle_structs.hpp>
namespace pumipic {

  template <class DataTypes, typename MemSpace>
  class DPS;

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class DPS_Input {
  public:
    typedef typename ParticleStructure<DataTypes, MemSpace>::kkLidView kkLidView;
    typedef typename ParticleStructure<DataTypes, MemSpace>::kkGidView kkGidView;
    typedef typename ParticleStructure<DataTypes, MemSpace>::MTVs MTVs;
    typedef Kokkos::TeamPolicy<typename MemSpace::execution_space> PolicyType;
    DPS_Input(PolicyType& p, lid_t num_elements,
              lid_t num_particles, kkLidView particles_per_elements, kkGidView element_gids,
              kkLidView particle_elements = kkLidView(), MTVs particle_info = NULL, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    //Extra padding at the end of the structure to allow growth [default = 0.05 (5%)]
    double extra_padding;

    //String identification for the particle structure
    std::string name;

    friend class DPS<DataTypes, MemSpace>;
  protected:
    PolicyType policy;
    lid_t ne, np;
    kkLidView ppe;
    kkGidView e_gids;
    kkLidView particle_elms;
    MTVs p_info;
    MPI_Comm mpi_comm;
  };

  template <class DataTypes, typename MemSpace>
  DPS_Input<DataTypes, MemSpace>::DPS_Input(PolicyType& p, lid_t ne_,
                                            lid_t np_, kkLidView ppe_, kkGidView eg,
                                            kkLidView pes, MTVs info, MPI_Comm comm) :
    policy(p), ne(ne_), np(np_), ppe(ppe_), e_gids(eg),
    particle_elms(pes), p_info(info), mpi_comm(comm) {
    extra_padding = 0.05;
    name = "ptcls";
  }
}
