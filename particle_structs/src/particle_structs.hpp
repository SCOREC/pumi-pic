#pragma once

#include "particle_structure.hpp"
#include "scs/SellCSigma.hpp"

namespace particle_structs {
  ParticleStructure* buildSCS(PolicyType& p,
                              lid_t sigma, lid_t vertical_chunk_size,
                              lid_t num_elements, lid_t num_particles,
                              kkLidView particles_per_element, kkGidView element_gids,
                              kkLidView particle_elements = kkLidView(),
                              MemberTypeViews<DataTypes> particle_info = NULL);
}
