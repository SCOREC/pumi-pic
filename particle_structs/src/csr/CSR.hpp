#pragma once

#include <particle_structure.hpp>
namespace particle_structs {
  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CSR : public ParticleStructure<DataTypes, MemSpace> {
  public:
    using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
    using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
    using typename ParticleStructure<DataTypes, MemSpace>::device_type;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::MTVs;

    CSR() = delete;
    CSR(const CSR&) = delete;
    CSR& operator=(const CSR&) = delete;

    CSR(lid_t num_elements, lid_t num_particles, kkLidView particles_per_element,
        kkGidView element_gids, kkLidView particle_elements = kkLidView(),
        MTVs particle_info = NULL);
    ~CSR();

    //Functions from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::nElems;
    using ParticleStructure<DataTypes, MemSpace>::nPtcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity;
    using ParticleStructure<DataTypes, MemSpace>::numRows;


    void migrate(kkLidView new_element, kkLidView new_process,
                 kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particle_info = NULL);

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL);

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string s="");

    void printMetrics() const;

  private:
    //Variables from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::num_elems;
    using ParticleStructure<DataTypes, MemSpace>::num_ptcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity_;
    using ParticleStructure<DataTypes, MemSpace>::num_rows;
    using ParticleStructure<DataTypes, MemSpace>::ptcl_data;
    using ParticleStructure<DataTypes, MemSpace>::num_types;

    //Offsets array into CSR
    kkLidView offsets;
  };

  template <class DataTypes, typename MemSpace>
  CSR<DataTypes, MemSpace>::CSR(lid_t num_elements, lid_t num_particles,
                                kkLidView particles_per_element,
                                kkGidView element_gids,
                                kkLidView particle_elements,
                                MTVs particle_info) {

  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
               kkLidView new_particle_elements,
               MTVs new_particle_info) {

  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {

  }

  template <class DataTypes, typename MemSpace>
  template <typename FunctionType>
  void CSR<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string s) {

  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::printMetrics() const {

  }
}
