#pragma once
#include <particle_structs.hpp>

namespace pumipic{

  template<class DataTypes, typename MemSpace>
  class CSR;

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CSR_Input{
  public:
    typedef typename ParticleStructure<DataTypes, MemSpace>::kkLidView kkLidView;
    typedef typename ParticleStructure<DataTypes, MemSpace>::kkGidView kkGidView;
    typedef typename ParticleStructure<DataTypes, MemSpace>::MTVs MTVs;
    typedef Kokkos::TeamPolicy<typename MemSpace::execution_space> PolicyType;
    
    CSR_Input(PolicyType& p, lid_t num_elements, lid_t num_particles, kkLidView particles_per_element, 
        kkGidView element_gids, kkLidView particle_elements = kkLidView(), MTVs particle_info = NULL, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    //Whether to reallocate the ptcl_structure on every rebuild or only when size requires
    bool always_realloc = false;

    //Reallocate structure when num_ptcls < minimize_size*capacity_
    double minimize_size = 0.8;

    //Amount of padding beyond the number of particles
    double padding_amount = 1.05; //1.05*num_ptcls

    std::string name;

    friend class CSR<DataTypes, MemSpace>;

  protected:
    PolicyType policy;
    lid_t ne, np;
    kkLidView ppe;
    kkGidView e_gids;
    kkLidView particle_elems;
    MTVs p_info;
    MPI_Comm mpi_comm;

  }; //end class CSR_Input

  template <class DataTypes, typename MemSpace>
  CSR_Input<DataTypes, MemSpace>::CSR_Input(PolicyType& p, lid_t num_elements, lid_t num_particles, 
      kkLidView particles_per_element, kkGidView element_gids, kkLidView particle_elements, MTVs particle_info, MPI_Comm comm) :
    policy(p), ne(num_elements), np(num_particles), ppe(particles_per_element), 
    e_gids(element_gids), particle_elems(particle_elements), p_info(particle_info), mpi_comm(comm)
  {
    name = "ptcls";
  }


}//end namespace pumipic
