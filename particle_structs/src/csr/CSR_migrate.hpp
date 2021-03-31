#pragma once 

namespace pumipic {
  
  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    //temporary single node implementation to allow running 
    //of GITRm on a single node (calls migrate only which calls
    //rebuild later on
    //const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("csr_migrate");
    Kokkos::Timer timer;

    //Distributor size & rank for performing migration
    int comm_size = dist.num_ranks();
    int comm_rank;
    MPI_Comm_rank(dist.mpi_comm(), &comm_rank);

    //If serial, skip migration
    if (comm_size == 1) {
      rebuild(new_element, new_particle_elements, new_particle_info);
      RecordTime("CSR particle migration", timer.seconds()/*, btime*/);
      Kokkos::Profiling::popRegion();
      return;
    }

    Kokkos::Profiling::popRegion();
  }

} //end namespace pumipic
