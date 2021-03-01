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
  void CabM<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    /// @todo add prebarrier to main ParticleStructure files
    //const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("cabm_migrate");
    Kokkos::Timer timer;

    //Distributor size & rank for performing migration
    int comm_size = dist.num_ranks();
    int comm_rank;
    MPI_Comm_rank(dist.mpi_comm(), &comm_rank);

    //If serial, skip migration
    if (comm_size == 1) {
      rebuild(new_element, new_particle_elements, new_particle_info);
      //RecordTime("CabM particle migration", timer.seconds(), btime);
      RecordTime("CabM particle migration", timer.seconds());
      Kokkos::Profiling::popRegion();
      return;
    }

    //Count number of particles to send to each process
    kkLidView num_send_particles("num_send_particles", comm_size + 1);
    auto count_sending_particles = PS_LAMBDA(lid_t element_id, lid_t particle_id, bool mask) {
      const lid_t process = new_process(particle_id);
      const lid_t process_index = dist.index(process);
      Kokkos::atomic_fetch_add(&(num_send_particles(process_index)),
                               mask * (process != comm_rank));
    };
    parallel_for(count_sending_particles);

    /********* Send # of particles being sent to each process *********/
    kkLidView num_recv_particles("num_recv_particles", comm_size + 1);
    int num_send_ranks = dist.isWorld() ? 0 : comm_size - 1;
    MPI_Request* count_send_requests = NULL;
    if (num_send_ranks > 0)
      count_send_requests = new MPI_Request[num_send_ranks];
    int num_recv_ranks = dist.isWorld() ? 1 : comm_size - 1;
    MPI_Request* count_recv_requests = new MPI_Request[num_recv_ranks];
    if (dist.isWorld())
      PS_Comm_Ialltoall(num_send_particles, 1, num_recv_particles, 1,
                        dist.mpi_comm(), count_recv_requests);
    else {
      int request_index = 0;
      for (int i = 0; i < comm_size; ++i) {
        int rank = dist.rank_host(i);
        if (rank != comm_rank) {
          PS_Comm_Isend(num_send_particles, i, 1, rank, 0, dist.mpi_comm(),
                        count_send_requests + request_index);
          PS_Comm_Irecv(num_recv_particles, i, 1, rank, 0, dist.mpi_comm(),
                        count_recv_requests + request_index);
          ++request_index;
        }
      }
    }

    //Gather sending particle data
    //Perform an ex-sum on num_send_particles & num_recv_particles
    kkLidView offset_send_particles("offset_send_particles", comm_size+1);
    kkLidView offset_send_particles_temp("offset_send_particles_temp", comm_size + 1);
    exclusive_scan(num_send_particles, offset_send_particles);
    Kokkos::deep_copy(offset_send_particles_temp, offset_send_particles);
    kkLidHostMirror offset_send_particles_host = deviceToHost(offset_send_particles);

    //Create arrays for particles being sent
    lid_t np_send = offset_send_particles_host(comm_size);
    kkLidView send_element("send_element", np_send);
    MTVs send_particle;
    //Allocate views for each data type into send_particle[type]
    CreateViews<device_type, DataTypes>(send_particle, np_send);
    kkLidView send_index("send_particle_index", capacity());
    auto element_to_gid_local = element_to_gid;
    auto gatherParticlesToSend = PS_LAMBDA(lid_t element_id, lid_t particle_id, lid_t mask) {
      const lid_t process = new_process(particle_id);
      const lid_t process_index = dist.index(process);
      if (mask && process != comm_rank) {
        send_index(particle_id) =
          Kokkos::atomic_fetch_add(&(offset_send_particles_temp(process_index)),1);
        const lid_t index = send_index(particle_id);
        send_element(index) = element_to_gid_local(new_element(particle_id));
      }
    };
    parallel_for(gatherParticlesToSend);
    //Copy the values from ptcl_data[type][particle_id] into send_particle[type](index) for each data type
    CopyParticlesToSendFromAoSoA<CabM<DataTypes, MemSpace>, DataTypes>(this, send_particle,
                                                                    aosoa_,
                                                                    new_process,
                                                                    send_index);

    /// @todo finish

    //RecordTime("CabM particle migration", timer.seconds(), btime);
    RecordTime("CabM particle migration", timer.seconds());

    Kokkos::Profiling::popRegion();
  }

}