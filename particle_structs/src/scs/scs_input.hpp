#pragma once
#include <particle_structs.hpp>
namespace pumipic {
    enum PaddingStrategy {
      //Divide all padding evenly to each element [Default]
      PAD_EVENLY,
      //Divide padding proportionally (more particles in element = more padding)
      PAD_PROPORTIONALLY,
      //Divide padding inverse-proportionally (more particles in element = less padding)
      PAD_INVERSELY
    };
  template <class DataTypes, typename MemSpace>
  class SellCSigma;

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class SCS_Input {
  public:
    typedef typename ParticleStructure<DataTypes, MemSpace>::kkLidView kkLidView;
    typedef typename ParticleStructure<DataTypes, MemSpace>::kkGidView kkGidView;
    typedef typename ParticleStructure<DataTypes, MemSpace>::MTVs MTVs;
    typedef Kokkos::TeamPolicy<typename MemSpace::execution_space> PolicyType;
    SCS_Input(PolicyType& p, lid_t sigma, lid_t vertical_chunk_size, lid_t num_elements,
              lid_t num_particles, kkLidView particles_per_elements, kkGidView element_gids,
              kkLidView particle_elements = kkLidView(), MTVs particle_info = NULL);

    //Percent padding to add based on the padding strategy [default = 0.1 (10%)]
    double shuffle_padding;
    //Extra padding at the end of the structure to allow growth [default = 0.05 (5%)]
    double extra_padding;

    //Padding strategy
    PaddingStrategy padding_strat;

    //String identification for the particle structure
    std::string name;

    friend class SellCSigma<DataTypes, MemSpace>;
  protected:
    PolicyType policy;
    lid_t sig, V;
    lid_t ne, np;
    kkLidView ppe;
    kkGidView e_gids;
    kkLidView particle_elms;
    MTVs p_info;
  };

  template <class DataTypes, typename MemSpace>
  SCS_Input<DataTypes, MemSpace>::SCS_Input(PolicyType& p, lid_t sigma, lid_t V_, lid_t ne_,
                                            lid_t np_, kkLidView ppe_, kkGidView eg,
                                            kkLidView pes, MTVs info) :
    policy(p), sig(sigma), V(V_), ne(ne_), np(np_), ppe(ppe_), e_gids(eg),
    particle_elms(pes), p_info(info) {
    shuffle_padding = 0.1;
    extra_padding = 0.05;
    padding_strat = PAD_EVENLY;
    name = "ptcls";
  }
}
