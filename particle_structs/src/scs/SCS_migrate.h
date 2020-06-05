#pragma once
#include <psMemberType.h>
namespace pumipic {

  template<class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                                  Distributor<MemSpace> dist,
                                                  kkLidView new_particle_elements,
                                                  MTVs new_particle_info) {
    const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("scs_migrate");
    Kokkos::Timer timer;
    /********* Send # of particles being sent to each process *********/
    int comm_size = dist.num_ranks();
    int comm_rank;
    MPI_Comm_rank(dist.mpi_comm(), &comm_rank);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (comm_size == 1) {
      rebuild(new_element, new_particle_elements, new_particle_info);
      if(!world_rank || world_rank == world_size/2)
        fprintf(stderr, "%d ps particle migration (seconds) %f\n", world_rank, timer.seconds());
      Kokkos::Profiling::popRegion();
      return;
    }
    Kokkos::Timer alltoall;
    kkLidView num_send_particles("num_send_particles", comm_size + 1);
    auto count_sending_particles = PS_LAMBDA(lid_t element_id, lid_t particle_id, bool mask) {
      const lid_t process = new_process(particle_id);
      const lid_t process_index = dist.index(process);
      Kokkos::atomic_fetch_add(&(num_send_particles(process_index)),
                               mask * (process != comm_rank));
    };
    parallel_for(count_sending_particles);
    kkLidView num_recv_particles("num_recv_particles", comm_size + 1);
    if (dist.isWorld())
      PS_Comm_Alltoall(num_send_particles, 1, num_recv_particles, 1, dist.mpi_comm());
    else {
      MPI_Request* count_send_requests = new MPI_Request[comm_size - 1];
      MPI_Request* count_recv_requests = new MPI_Request[comm_size - 1];
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
      PS_Comm_Waitall<device_type>(comm_size - 1, count_send_requests, MPI_STATUSES_IGNORE);
      PS_Comm_Waitall<device_type>(comm_size - 1, count_recv_requests, MPI_STATUSES_IGNORE);
      delete [] count_send_requests;
      delete [] count_recv_requests;
    }
    if (world_rank == world_size/2) {
      fprintf(stderr, "%d AllToAll %f\n", world_rank, alltoall.seconds());
    }
    Kokkos::Timer countTotals;
    lid_t num_sending_to = 0, num_receiving_from = 0;
    Kokkos::parallel_reduce("sum_senders", comm_size, KOKKOS_LAMBDA (const lid_t& i, lid_t& lsum ) {
        lsum += (num_send_particles(i) > 0);
      }, num_sending_to);
    Kokkos::parallel_reduce("sum_receivers", comm_size, KOKKOS_LAMBDA (const lid_t& i, lid_t& lsum ) {
        lsum += (num_recv_particles(i) > 0);
      }, num_receiving_from);

    if (num_sending_to == 0 && num_receiving_from == 0) {
      rebuild(new_element, new_particle_elements, new_particle_info);
      if(!world_rank || world_rank == world_size/2)
        fprintf(stderr, "%d ps particle migration (seconds) %f\n", world_rank, timer.seconds());
      Kokkos::Profiling::popRegion();
      return;
    }
    if (world_rank == world_size/2) {
      fprintf(stderr, "%d countTotals %f\n", world_rank, countTotals.seconds());
    }

    Kokkos::Timer gatherSend;
    /********** Send particle information to new processes **********/
    //Perform an ex-sum on num_send_particles & num_recv_particles
    kkLidView offset_send_particles("offset_send_particles", comm_size+1);
    kkLidView offset_send_particles_temp("offset_send_particles_temp", comm_size + 1);
    kkLidView offset_recv_particles("offset_recv_particles", comm_size+1);
    exclusive_scan(num_send_particles, offset_send_particles);
    Kokkos::deep_copy(offset_send_particles_temp, offset_send_particles);

    exclusive_scan(num_recv_particles, offset_recv_particles);
    kkLidHostMirror offset_send_particles_host = deviceToHost(offset_send_particles);
    kkLidHostMirror offset_recv_particles_host = deviceToHost(offset_recv_particles);

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
    CopyParticlesToSend<SellCSigma<DataTypes, MemSpace>, DataTypes>(this, send_particle,
                                                                    ptcl_data,
                                                                    new_process,
                                                                    send_index);

    if (world_rank == world_size/2) {
      fprintf(stderr, "%d gatherSend %f\n", world_rank, gatherSend.seconds());
    }

    Kokkos::Timer allocate;
    //Create arrays for particles being received
    lid_t new_ptcls = new_particle_elements.size();
    lid_t np_recv = offset_recv_particles_host(comm_size);
    kkLidView recv_element("recv_element", np_recv + new_ptcls);
    MTVs recv_particle;
    //Allocate views for each data type into recv_particle[type]
    CreateViews<device_type, DataTypes>(recv_particle, np_recv + new_ptcls);

    if (world_rank == world_size/2) {
      fprintf(stderr, "%d allocate %f\n", world_rank, allocate.seconds());
    }

    Kokkos::Timer postComms;

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

    if (world_rank == world_size/2) {
      fprintf(stderr, "%d postComms %f\n", world_rank, postComms.seconds());
    }

    Kokkos::Timer waiting;
    PS_Comm_Waitall<device_type>(num_recvs, recv_requests, MPI_STATUSES_IGNORE);
    delete [] recv_requests;

    if (world_rank == world_size/2) {
      fprintf(stderr, "%d waiting %f\n", world_rank, waiting.seconds());
    }

    Kokkos::Timer finalSetup;

    /********** Convert the received element from element gid to element lid *********/
    auto element_gid_to_lid_local = element_gid_to_lid;
    Kokkos::parallel_for(np_recv, KOKKOS_LAMBDA(const lid_t& i) {
        const gid_t gid = recv_element(i);
        const lid_t index = element_gid_to_lid_local.find(gid);
        recv_element(i) = element_gid_to_lid_local.value_at(index);
      });

    /********** Set particles that were sent to non existent on this process *********/
    auto removeSentParticles = PS_LAMBDA(lid_t element_id, lid_t particle_id, lid_t mask) {
      const bool sent = new_process(particle_id) != comm_rank;
      const lid_t elm = new_element(particle_id);
      //Subtract (its value + 1) to get to -1 if it was sent, 0 otherwise
      new_element(particle_id) -= (elm + 1) * sent;
    };
    parallel_for(removeSentParticles);

    /********** Add new particles to the migrated particles *********/
    kkLidView new_ptcl_map("new_ptcl_map", new_ptcls);
    Kokkos::parallel_for(new_ptcls, KOKKOS_LAMBDA(const lid_t& i) {
        recv_element(np_recv + i) = new_particle_elements(i);
        new_ptcl_map(i) = np_recv + i;
    });
    CopyViewsToViews<kkLidView, DataTypes>(recv_particle, new_particle_info, new_ptcl_map);

    if (world_rank == world_size/2) {
      fprintf(stderr, "%d finalSetup %f\n", world_rank, finalSetup.seconds());
    }

    /********** Combine and shift particles to their new destination **********/
    rebuild(new_element, recv_element, recv_particle);

    Kokkos::Timer cleanup;
    //Cleanup
    PS_Comm_Waitall<device_type>(num_sends, send_requests, MPI_STATUSES_IGNORE);
    delete [] send_requests;
    destroyViews<DataTypes, memory_space>(send_particle);
    destroyViews<DataTypes, memory_space>(recv_particle);

    if (world_rank == world_size/2) {
      fprintf(stderr, "%d cleanup %f\n", world_rank, cleanup.seconds());
    }

    if(!world_rank || world_rank == world_size/2)
      fprintf(stderr, "%d ps particle migration (seconds) %f pre-barrier (seconds) %f\n",
              world_rank, timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }
}
