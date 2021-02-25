#pragma once

namespace pumipic {
    
  /**
   *
   * @param[in] new_element
   * @param[in] new_process
   * @param[in] dist
   * @param[in] new_particle_elements
   * @param[in] new_particle_info
  */
  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    /// @todo implement migrate
    fprintf(stderr, "[WARNING] CabM migrate(...) not implemented\n");
  }

}