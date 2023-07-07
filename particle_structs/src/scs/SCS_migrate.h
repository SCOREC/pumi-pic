#pragma once
namespace pumipic {

  template<class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                                  Distributor<MemSpace> dist,
                                                  kkLidView new_particle_elements,
                                                  MTVs new_particle_info) {

    const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("scs_migrate");
    Kokkos::Timer timer;

    //Distributor size & rank for performing migration
    int comm_size = dist.num_ranks();
    int comm_rank;
    MPI_Comm_rank(dist.mpi_comm(), &comm_rank);

    //If serial, skip migration
    if (comm_size == 1) {
      RecordTime(name + " particle migration", timer.seconds(), btime);
      rebuild(new_element, new_particle_elements, new_particle_info);
      Kokkos::Profiling::popRegion();
      return;
    }


    //Count number of particles to send to each process
    kkLidView num_send_particles("num_send_particles", comm_size + 1);
    auto count_sending_particles = PS_LAMBDA(const lid_t& element_id, const lid_t& particle_id, const bool& mask) {
      const lid_t process = new_process(particle_id);
      if (mask && (process != comm_rank)) {
        const lid_t process_index = dist.index(process);
        Kokkos::atomic_increment(&num_send_particles(process_index));
      }
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
    if (dist.isWorld()) {
#ifdef PP_USE_GPU
      PS_Comm_Alltoall(num_send_particles, 1, num_recv_particles, 1, dist.mpi_comm());
#else //Causes test to fail on Frontier     
      PS_Comm_Ialltoall(num_send_particles, 1, num_recv_particles, 1, dist.mpi_comm(), count_recv_requests);
#endif
    }
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
    kkLidView offset_send_particles_temp(Kokkos::ViewAllocateWithoutInitializing("offset_send_particles_temp"), comm_size + 1);
    exclusive_scan(num_send_particles, offset_send_particles, execution_space());
    Kokkos::deep_copy(offset_send_particles_temp, offset_send_particles);
    kkLidHostMirror offset_send_particles_host = deviceToHost(offset_send_particles);

    //Create arrays for particles being sent
    lid_t np_send = offset_send_particles_host(comm_size);
    kkLidView send_element(Kokkos::ViewAllocateWithoutInitializing("send_element"), np_send);
    MTVs send_particle;
    //Allocate views for each data type into send_particle[type]
    CreateViews<device_type, DataTypes>(send_particle, np_send);
    kkLidView send_index(Kokkos::ViewAllocateWithoutInitializing("send_particle_index"), capacity());
    auto element_to_gid_local = element_to_gid;
    auto gatherParticlesToSend = PS_LAMBDA(const lid_t& element_id, const lid_t& particle_id, const bool& mask) {
      const lid_t process = new_process(particle_id);
      if (mask && process != comm_rank) {
        const lid_t process_index = dist.index(process);
        send_index(particle_id) =
          Kokkos::atomic_fetch_add(&(offset_send_particles_temp(process_index)),1);
        const lid_t index = send_index(particle_id);
        send_element(index) = element_to_gid_local(new_element(particle_id));
      }
    };
    parallel_for(gatherParticlesToSend);
    //Copy the values from ptcl_data[type][particle_id] into send_particle[type](index) for each data type
    CopyParticlesToSend<SellCSigma<DataTypes, MemSpace>, DataTypes>(this, send_particle,
                                                                    ptcl_data,
                                                                    new_process,
                                                                    send_index);
#ifdef PP_USE_GPU
    if (!dist.isWorld())
#endif //Causes test to fail on Frontier
      PS_Comm_Waitall<device_type>(num_recv_ranks, count_recv_requests, MPI_STATUSES_IGNORE);
    delete [] count_recv_requests;

    //Count the number of processes being sent to and recv from
    lid_t num_sending_to = 0, num_receiving_from = 0;
    Kokkos::parallel_reduce("sum_senders", comm_size,
                            KOKKOS_LAMBDA (const lid_t& i, lid_t& lsum ) {
      lsum += (num_send_particles(i) > 0);
    }, num_sending_to);
    Kokkos::parallel_reduce("sum_receivers", comm_size,
                            KOKKOS_LAMBDA (const lid_t& i, lid_t& lsum ) {
      lsum += (num_recv_particles(i) > 0);
    }, num_receiving_from);

    //wait for send requests if there are any
    if (count_send_requests) {
      PS_Comm_Waitall<device_type>(num_send_ranks, count_send_requests,
                                   MPI_STATUSES_IGNORE);
      delete [] count_send_requests;
    }

    //If no particles are being sent or received, perform rebuild
    if (num_sending_to == 0 && num_receiving_from == 0) {
      destroyViews<DataTypes, memory_space>(send_particle);
      rebuild(new_element, new_particle_elements, new_particle_info);
      RecordTime(name +" particle migration", timer.seconds(), btime);
      Kokkos::Profiling::popRegion();
      return;
    }


    //Offset the recv particles
    kkLidView offset_recv_particles("offset_recv_particles", comm_size+1);
    exclusive_scan(num_recv_particles, offset_recv_particles, execution_space());
    kkLidHostMirror offset_recv_particles_host = deviceToHost(offset_recv_particles);
    int np_recv = offset_recv_particles_host(comm_size);

    //Create arrays for particles being received
    lid_t new_ptcls = new_particle_elements.size();
    kkLidView recv_element(Kokkos::ViewAllocateWithoutInitializing("recv_element"), np_recv + new_ptcls);
    MTVs recv_particle;
    //Allocate views for each data type into recv_particle[type]
    CreateViews<device_type, DataTypes>(recv_particle, np_recv + new_ptcls);

    //Get pointers to the data for MPI calls
    lid_t send_num = 0, recv_num = 0;
    lid_t num_sends = num_sending_to * (num_types + 1);
    lid_t num_recvs = num_receiving_from * (num_types + 1);
    MPI_Request* send_requests = new MPI_Request[num_sends];
    MPI_Request* recv_requests = new MPI_Request[num_recvs];
    //Send the particles to each neighbor
    for (lid_t i = 0; i < comm_size; ++i) {
      int rank = dist.rank_host(i);
      if (rank == comm_rank)
        continue;

      //Sending
      lid_t num_send = offset_send_particles_host(i+1) - offset_send_particles_host(i);
      if (num_send > 0) {
        lid_t start_index = offset_send_particles_host(i);
        PS_Comm_Isend(send_element, start_index, num_send, rank, 0, dist.mpi_comm(),
                      send_requests +send_num);
        send_num++;
        SendViews<device_type, DataTypes>(send_particle, start_index, num_send, rank, 1,
                                          dist.mpi_comm(), send_requests + send_num);
        send_num+=num_types;
      }
      //Receiving
      lid_t num_recv = offset_recv_particles_host(i+1) - offset_recv_particles_host(i);
      if (num_recv > 0) {
        lid_t start_index = offset_recv_particles_host(i);
        PS_Comm_Irecv(recv_element, start_index, num_recv, rank, 0, dist.mpi_comm(),
                      recv_requests + recv_num);
        recv_num++;
        RecvViews<device_type, DataTypes>(recv_particle,start_index, num_recv, rank, 1,
                                          dist.mpi_comm(), recv_requests + recv_num);
        recv_num+=num_types;
      }
    }

    PS_Comm_Waitall<device_type>(num_recvs, recv_requests, MPI_STATUSES_IGNORE);
    delete [] recv_requests;

    /********** Convert the received element from element gid to element lid *********/
    auto element_gid_to_lid_local = element_gid_to_lid;
    Kokkos::parallel_for(np_recv, KOKKOS_LAMBDA(const lid_t& i) {
        const gid_t gid = recv_element(i);
        const lid_t index = element_gid_to_lid_local.find(gid);
        assert(element_gid_to_lid_local.valid_at(index));
        recv_element(i) = element_gid_to_lid_local.value_at(index);
      });

    /********** Set particles that were sent to non existent on this process *********/
    auto removeSentParticles = PS_LAMBDA(const lid_t& element_id, const lid_t& particle_id, const bool& mask) {
      const bool sent = new_process(particle_id) != comm_rank;
      const lid_t elm = new_element(particle_id);
      //Subtract (its value + 1) to get to -1 if it was sent, 0 otherwise
      new_element(particle_id) -= (elm + 1) * sent;
    };
    parallel_for(removeSentParticles);

    /********** Add new particles to the migrated particles *********/
    kkLidView new_ptcl_map(Kokkos::ViewAllocateWithoutInitializing("new_ptcl_map"), new_ptcls);
    Kokkos::parallel_for(new_ptcls, KOKKOS_LAMBDA(const lid_t& i) {
        recv_element(np_recv + i) = new_particle_elements(i);
        new_ptcl_map(i) = np_recv + i;
    });
    CopyViewsToViews<kkLidView, DataTypes>(recv_particle, new_particle_info, new_ptcl_map);


    /********** Combine and shift particles to their new destination **********/
    Kokkos::Timer rebuild_subtract;
    rebuild(new_element, recv_element, recv_particle);
    const auto temp = rebuild_subtract.seconds();

    //Cleanup
    PS_Comm_Waitall<device_type>(num_sends, send_requests, MPI_STATUSES_IGNORE);
    delete [] send_requests;
    destroyViews<DataTypes, memory_space>(send_particle);
    destroyViews<DataTypes, memory_space>(recv_particle);

    RecordTime(name +" particle migration", timer.seconds() - temp, btime);

    Kokkos::Profiling::popRegion();
  }
}
