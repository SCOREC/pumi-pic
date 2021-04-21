#pragma once

namespace pumipic {

  /**
   * Distributes current and new particles across a number of processes, then rebuilds
   * @param[in] new_element view of ints representing new elements for each current particle (-1 for removal)
   * @param[in] new_process view of ints representing new processes for each current particle
   * @param[in] dist Distributor set up for keeping track of processes
   * @param[in] new_particle_elements view of ints representing new elements for new particles (-1 for removal)
   * @param[in] new_particle_info array of views filled with particle data
  */
  template <class DataTypes, typename MemSpace>
  void DPS<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    fprintf(stderr, "[WARNING] migrate not yet implemented!\n");
  }

}