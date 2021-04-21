#pragma once

#include <ppTiming.hpp>

namespace pumipic {
  /**
   * Fully rebuild the AoSoA with these new parent SoAs and particles
   *     by copying into a new AoSoA and overwriting the old one.
   *     Delete particles with new_element(ptcl) < 0
   * @param[in] new_element view of ints with new elements for each particle
   * @param[in] new_particle_elements view of ints, representing which elements
   *    particle reside in
   * @param[in] new_particles array of views filled with particle data
   * @exception new_particle_elements(ptcl) < 0,
   *    undefined behavior for new_particle_elements.size() != sizeof(new_particles),
   *    undefined behavior for numberoftypes(new_particles) != numberoftypes(DataTypes)
   *    undefined behavior for new_element(ptcl) >= num_elms or new_particle_elements(ptcl) >= num_elems
  */
  template <class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {
    fprintf(stderr, "[WARNING] rebuild not yet implemented!\n");
  }

}