#pragma once
#include <particle_structs.hpp>
namespace pumipic {

  template <class DataTypes, typename MemSpace>
  class CabM;

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CabM_Input {
  public:
    typedef typename ParticleStructure<DataTypes, MemSpace>::kkLidView kkLidView;
    typedef typename ParticleStructure<DataTypes, MemSpace>::kkGidView kkGidView;
    typedef typename ParticleStructure<DataTypes, MemSpace>::MTVs MTVs;
    typedef Kokkos::TeamPolicy<typename MemSpace::execution_space> PolicyType;
    CabM_Input(PolicyType& p, lid_t num_elements,
              lid_t num_particles, kkLidView particles_per_elements, kkGidView element_gids,
              bool use_swap_ = false, kkLidView particle_elements = kkLidView(), MTVs particle_info = NULL);

    //Extra padding at the end of the structure to allow growth [default = 0.05 (5%)]
    double extra_padding;

    //String identification for the particle structure
    std::string name;

    friend class CabM<DataTypes, MemSpace>;
  protected:
    PolicyType policy;
    lid_t ne, np;
    kkLidView ppe;
    kkGidView e_gids;
    kkLidView particle_elms;
    MTVs p_info;
    bool use_swap;
  };

  template <class DataTypes, typename MemSpace>
  CabM_Input<DataTypes, MemSpace>::CabM_Input(PolicyType& p, lid_t ne_,
                                            lid_t np_, kkLidView ppe_, kkGidView eg,
                                            bool use_swap_, kkLidView pes, MTVs info) :
    policy(p), ne(ne_), np(np_), ppe(ppe_), e_gids(eg),
    use_swap(use_swap_), particle_elms(pes), p_info(info) {
    extra_padding = 0.05;
    name = "ptcls";
  }
}
