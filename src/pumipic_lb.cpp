#include "pumipic_lb.hpp"

#include "pumipic_mesh.hpp"
#include <particle_structs.hpp>
#include <Omega_h_for.hpp>

namespace pumipic {
  typedef Omega_h::LO LO;
  void prepareSafeCommArray(int ne, Omega_h::LOs comm_indices, Omega_h::LOs safeTag,
                            Omega_h::Write<LO> safe_comm_array) {
    auto setSafeCommArray = OMEGA_H_LAMBDA(const LO elm) {
      const LO index = comm_indices[elm];
      safe_comm_array[index] = safeTag[elm];
    };
    Omega_h::parallel_for(ne, setSafeCommArray, "setSafeCommArray");
  }

  ParticleBalancer::~ParticleBalancer() {
    agi::destroyGraph(weightGraph);
    //Return PCU communicator to world
    PCU_Switch_Comm(MPI_COMM_WORLD);
  }
  ParticleBalancer::ParticleBalancer(Mesh& picparts) {
    Omega_h::CommPtr comm = picparts.comm();
    //Change PCU communicator to the mesh communicator
    PCU_Switch_Comm(comm->get_impl());
    int comm_rank = comm->rank();
    //Prepare comm array of safe values for core regions
    int dim = picparts.dim();
    int ne = picparts->nelems();
    Omega_h::LOs offsets_nents = picparts.nentsOffsets(dim);
    Omega_h::LOs comm_indices = picparts.commArrayIndex(dim);
    Omega_h::Write<LO> safe_comm_array(ne, 0);
    prepareSafeCommArray(ne, comm_indices, picparts.safeTag(), safe_comm_array);

    //Communicate safe zone for each element of core region (GPU/CPU)
    Omega_h::HostWrite<LO> safe_comm_array_host(safe_comm_array);
    Omega_h::HostRead<LO> offsets_nents_host(offsets_nents);
    int nbuffers = picparts.numBuffers(dim) - 1;
    Omega_h::HostWrite<LO> buffer_ranks = picparts.bufferedRanks(dim);
    LO core_nents = offsets_nents_host[comm_rank + 1] - offsets_nents_host[comm_rank];
    int safe_buffer_size = core_nents * nbuffers;
    Omega_h::HostWrite<LO> safe_core_per_buffer(safe_buffer_size);
    MPI_Datatype bufferStride;
    MPI_Type_vector(core_nents, 1, nbuffers, MPI_INT, &bufferStride);
    MPI_Type_commit(&bufferStride);
    MPI_Request* send_requests = new MPI_Request[nbuffers];
    MPI_Request* recv_requests = new MPI_Request[nbuffers];
    for (int i = 0; i < nbuffers; ++i) {
      int buffer_rank = buffer_ranks[i];
      MPI_Irecv(safe_core_per_buffer.data() + i, core_nents, bufferStride,
                buffer_rank, 0, comm->get_impl(), recv_requests + i);

      int offset = offsets_nents_host[buffer_rank];
      int nents = offsets_nents_host[buffer_rank+1] - offsets_nents_host[buffer_rank];
      MPI_Isend(safe_comm_array_host.data() + offset, nents, MPI_INT,
                buffer_rank, 0, comm->get_impl(), send_requests + i);
    }
    MPI_Waitall(nbuffers, recv_requests, MPI_STATUSES_IGNORE);
    delete [] recv_requests;

    MPI_Waitall(nbuffers, send_requests, MPI_STATUSES_IGNORE);
    delete [] send_requests;

    //Determine the overlapping safe zones(sbars) for each element (CPU)
    Omega_h::HostWrite<int> core_elm_sbar = buildLocalSbarMap(comm_rank, core_nents,
                                                              buffer_ranks,
                                                              safe_core_per_buffer);

    sendCoreSbars(comm, buffer_ranks);

    //Create indexing of each sbar and the nodes in the Ngraph (CPU)
    std::unordered_map<int, int> sbar_local_to_global;
    globalNumberSbars(comm, sbar_local_to_global);
    cleanSbars(comm_rank);

    //Number Elements with global sbar ids
    numberElements(picparts, core_elm_sbar, sbar_local_to_global);

    //Build N-graph from indices (CPU)
    buildNgraph(comm);
    MPI_Type_free(&bufferStride);
  }

  ParticleBalancer::SBarUnmap::iterator ParticleBalancer::insert(Parts& p) {
    auto itr = sbar_ids.find(p);
    if (itr == sbar_ids.end()) {
      itr = (sbar_ids.insert(std::make_pair(p, sbar_ids.size()))).first;
    }
    return itr;
  }

  Omega_h::HostWrite<int> ParticleBalancer::buildLocalSbarMap(int comm_rank, int nelms,
                                           Omega_h::HostWrite<LO> buffer_ranks,
                                           Omega_h::HostWrite<LO> safe_core_per_buffer) {
    Omega_h::HostWrite<int> core_elm_sbar = Omega_h::HostWrite<int>(nelms, "core_sbars");
    int nbuffers = buffer_ranks.size();
    for (int i = 0; i < nelms; ++i) {
      Parts parts;
      parts.insert(comm_rank);
      for (int j = 0; j < nbuffers; ++j) {
        int index = i * nbuffers + j;
        if (safe_core_per_buffer[index])
          parts.insert(buffer_ranks[j]);
      }
      auto itr = insert(parts);
      core_elm_sbar[i] = itr->second;
    }
    return core_elm_sbar;
  }

  void ParticleBalancer::sendCoreSbars(Omega_h::CommPtr comm,
                                       Omega_h::HostWrite<LO> buffer_ranks) {
    int comm_rank = comm->rank();

    //Calculate message length
    int sbar_length = 0;
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); ++itr) {
      sbar_length += 1 + itr->first.size();
    }

    //Communicate message length
    int nbuffers = buffer_ranks.size();
    MPI_Request* send_requests = new MPI_Request[nbuffers];
    MPI_Request* recv_requests = new MPI_Request[nbuffers];
    int* message_lengths = new int[nbuffers];
    for (int i = 0; i < buffer_ranks.size(); ++i) {
      MPI_Irecv(message_lengths + i, 1, MPI_INT, buffer_ranks[i], 0,
                comm->get_impl(), recv_requests + i);
      MPI_Isend(&sbar_length, 1, MPI_INT, buffer_ranks[i], 0,
                comm->get_impl(), send_requests + i);
    }

    //Build message
    int* sbar_message = new int[sbar_length];
    int index = 0;
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); ++itr) {
      sbar_message[index++] = itr->first.size();
      for (auto itr2 = itr->first.begin(); itr2 != itr->first.end(); ++itr2) {
        sbar_message[index++] = *itr2;
      }
    }
    MPI_Waitall(nbuffers, recv_requests, MPI_STATUSES_IGNORE);

    //Build recv message block
    int total_message_size = 0;
    int* message_offsets = new int[nbuffers + 1];
    message_offsets[0] = 0;
    for (int i = 0; i < nbuffers; ++i) {
      total_message_size += message_lengths[i];
      message_offsets[i + 1] = total_message_size;
    }
    delete [] message_lengths; message_lengths = NULL;
    int* message = new int[total_message_size];
    MPI_Waitall(nbuffers, send_requests, MPI_STATUSES_IGNORE);

    //Send message
    for (int i = 0; i < buffer_ranks.size(); ++i) {
      int offset = message_offsets[i];
      int size = message_offsets[i+1] - message_offsets[i];
      MPI_Irecv(message + offset, size, MPI_INT, buffer_ranks[i], 0,
                comm->get_impl(), recv_requests + i);
      MPI_Isend(sbar_message, sbar_length, MPI_INT, buffer_ranks[i], 0,
                comm->get_impl(), send_requests + i);
    }

    MPI_Waitall(nbuffers, recv_requests, MPI_STATUSES_IGNORE);
    delete [] recv_requests;

    for (int i = 0; i < total_message_size; i += message[i] + 1) {
      int nparts = message[i];
      Parts parts;
      for (int j = i+1; j <= i + nparts; j++)
        parts.insert(message[j]);
      insert(parts);
    }
    //Cleanup
    delete [] sbar_message;
    delete [] message_offsets;
    delete [] message;
    MPI_Waitall(nbuffers, send_requests, MPI_STATUSES_IGNORE);
    delete [] send_requests;
  }

  //The process whose id is the smallest in an sbar is its "owner"
  //  and is in charge of numbering it
  void ParticleBalancer::globalNumberSbars(Omega_h::CommPtr comm,
                                           std::unordered_map<int,int>& sbar_local_to_global) {
    Parts sbar_sends, sbar_recvs;

    //Count how many sbars the process is numbering
    int comm_rank = comm->rank();
    int num_owned_sbar = 0;
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); ++itr) {
      if (*(itr->first.begin()) == comm_rank) {
        num_owned_sbar += itr->first.size();
      }
    }

    //Exclusive scan over number of counted sbars
    int start_num_sbar = 0;
    MPI_Exscan(&num_owned_sbar, &start_num_sbar, 1, MPI_INT, MPI_SUM, comm->get_impl());


    //Number all sbars which the process "owns"
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); ++itr) {
      if (*(itr->first.begin()) == comm_rank) {
        sbar_local_to_global[itr->second] = start_num_sbar;
        start_num_sbar += itr->first.size();
        auto itr2 = itr->first.begin();
        ++itr2;
        for (; itr2 != itr->first.end(); ++itr2)
          sbar_sends.insert(*itr2);
      }
      else {
        if (itr->first.find(comm_rank) != itr->first.end())
          sbar_recvs.insert(*(itr->first.begin()));
      }
    }
    int comm_size;
    MPI_Comm_size(comm->get_impl(), &comm_size);
    max_sbar = start_num_sbar;
    MPI_Bcast(&max_sbar, 1, MPI_INT, comm_size - 1, comm->get_impl());

    //Communicate local numbers to all buffers

    //Calculate message length
    std::unordered_map<int, int> send_lengths;
    std::unordered_map<int, int> recv_lengths;
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); itr++) {
      if (*(itr->first.begin()) == comm_rank) {
        for (auto pitr = itr->first.begin(); pitr != itr->first.end(); ++pitr) {
          if (*pitr != comm_rank)
            send_lengths[*pitr] += 2 + itr->first.size();
        }
      }
    }

    //Communicate message lengths
    MPI_Request* send_requests = new MPI_Request[sbar_sends.size()];
    MPI_Request* recv_requests = new MPI_Request[sbar_recvs.size()];
    int index = 0;
    for (auto itr = sbar_sends.begin(); itr != sbar_sends.end(); ++itr) {
      MPI_Isend(&(send_lengths[*itr]), 1, MPI_INT,
                *itr, 0, comm->get_impl(), send_requests + index);
      ++index;
    }
    index = 0;
    for (auto itr = sbar_recvs.begin(); itr != sbar_recvs.end(); ++itr) {
      recv_lengths[*itr] = 0;
      MPI_Irecv(&(recv_lengths[*itr]), 1, MPI_INT,
                *itr, 0, comm->get_impl(), recv_requests + index);
      ++index;
    }

    //Build message being sent
    std::unordered_map<int, int*> send_messages;
    std::unordered_map<int, int> send_index;
    for (auto itr = sbar_sends.begin(); itr != sbar_sends.end(); ++itr) {
      send_messages[*itr] = new int[send_lengths[*itr]];
      send_index[*itr] = 0;
    }
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); itr++) {
      int id = sbar_local_to_global[itr->second];
      if (*(itr->first.begin()) == comm_rank) {
        for (auto pitr = itr->first.begin(); pitr != itr->first.end(); ++pitr) {
          if (*pitr != comm_rank) {
            int& index = send_index[*pitr];
            int* arr = send_messages[*pitr];
            arr[index++] = itr->first.size();
            arr[index++] = id;
            for (auto pitr2 = itr->first.begin(); pitr2 != itr->first.end(); ++pitr2) {
              arr[index++] = *pitr2;
            }
          }
        }
      }
    }

    //Build memory for recv message
    MPI_Waitall(sbar_recvs.size(), recv_requests, MPI_STATUSES_IGNORE);
    std::unordered_map<int, int*> recv_messages;
    for (auto itr = sbar_recvs.begin(); itr != sbar_recvs.end(); ++itr) {
      recv_messages[*itr] = new int[recv_lengths[*itr]];
    }

    //Communicate messages
    MPI_Waitall(sbar_sends.size(), send_requests, MPI_STATUSES_IGNORE);

    index = 0;
    for (auto itr = sbar_sends.begin(); itr != sbar_sends.end(); ++itr) {
      MPI_Isend(send_messages[*itr], send_lengths[*itr], MPI_INT,
                *itr, 1, comm->get_impl(), send_requests + index);
      ++index;
    }
    index = 0;
    for (auto itr = sbar_recvs.begin(); itr != sbar_recvs.end(); ++itr) {
      MPI_Irecv(recv_messages[*itr], recv_lengths[*itr], MPI_INT,
                *itr, 1, comm->get_impl(), recv_requests + index);
      ++index;
    }

    MPI_Waitall(sbar_recvs.size(), recv_requests, MPI_STATUSES_IGNORE);
    delete [] recv_requests;
    for (auto itr = recv_messages.begin(); itr != recv_messages.end(); ++itr) {
      int* message = itr->second;
      for (int i = 0; i < recv_lengths[itr->first]; i += message[i] + 2) {
        int id = message[i+1];
        Parts res;
        for (int j = i + 2; j < i + 2 + message[i]; ++j) {
          res.insert(message[j]);
        }
        auto sbar_itr = sbar_ids.find(res);
        if (sbar_itr != sbar_ids.end()) {
          sbar_local_to_global[sbar_itr->second] = id;
        }
        else {
          char error[1024];
          char* ptr = error + sprintf(error, "Rank: %d Cannot find Sbar [", comm_rank);
          for (auto pitr = res.begin(); pitr != res.end(); ++pitr) {
            ptr += sprintf(ptr," %d", *pitr);
          }
          ptr += sprintf(ptr, "]\n");
          pPrintError( "%s", error);
        }
      }
    }

    MPI_Waitall(sbar_sends.size(), send_requests, MPI_STATUSES_IGNORE);
    delete [] send_requests;
    for (auto itr = send_messages.begin(); itr != send_messages.end(); ++itr)
      delete [] itr->second;
    for (auto itr = recv_messages.begin(); itr != recv_messages.end(); ++itr)
      delete [] itr->second;
  }

  void ParticleBalancer::cleanSbars(int comm_rank) {
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end();) {
      if (itr->first.find(comm_rank) == itr->first.end()) {
        itr = sbar_ids.erase(itr);
      }
      else
        ++itr;
    }
  }
  template <typename T, typename S>
  void buildMap(std::unordered_map<T, S>& map,
                Kokkos::UnorderedMap<T, S>& device_map) {
    Omega_h::HostWrite<T> keys_host(map.size(), "keys");
    Omega_h::HostWrite<S> values_host(map.size(), "values");
    int index = 0;
    for (auto itr = map.begin(); itr != map.end(); ++itr, ++index) {
      keys_host[index] = itr->first;
      values_host[index] = itr->second;
    }
    Omega_h::Write<T> keys(keys_host);
    Omega_h::Write<S> values(values_host);
    Omega_h::parallel_for(keys.size(), OMEGA_H_LAMBDA(const int i) {
      device_map.insert(keys[i], values[i]);
    });
  }
  void numberCore(int start, int nce, Kokkos::UnorderedMap<int, int>& map,
                  Omega_h::Write<Omega_h::LO> local_ids, Omega_h::Write<Omega_h::LO> elm_sbar) {
    auto numberCoreElms = OMEGA_H_LAMBDA(const int elm) {
      const int local_id = local_ids[elm];
      auto index = map.find(local_id);
      elm_sbar[elm + start] = map.value_at(index);
    };
    Omega_h::parallel_for(nce, numberCoreElms, "numberCoreElms");
  }
  Omega_h::LOs convertToMeshIndices(Omega_h::Write<LO> elem_sbars,
                                    Omega_h::LOs comm_indices) {
    Omega_h::Write<LO> elem_sbars_tag(elem_sbars.size());
    auto convertIndices = OMEGA_H_LAMBDA(const LO elm) {
      const LO comm_index = comm_indices[elm];
      elem_sbars_tag[elm] = elem_sbars[comm_index];
    };
    Omega_h::parallel_for(elem_sbars.size(), convertIndices, "convertIndices");
    return Omega_h::LOs(elem_sbars_tag);
  }

  void ParticleBalancer::numberElements(Mesh& picparts, Omega_h::HostWrite<int> local_sbar_host,
                                        std::unordered_map<int, int>& map) {
    Omega_h::CommPtr comm = picparts.comm();
    int comm_rank = comm->rank();
    int dim = picparts.dim();
    //Transfer information to the device
    Omega_h::Write<int> local_sbar(local_sbar_host);
    //Create UMap on device
    Kokkos::UnorderedMap<int, int> device_map(map.size());
    buildMap(map, device_map);

    //Global number core elements
    Omega_h::LOs offsets_nents = picparts.nentsOffsets(dim);
    Omega_h::HostRead<LO> offsets_nents_host = Omega_h::HostRead<LO>(offsets_nents);
    int start = offsets_nents_host[comm_rank];
    int nce =  offsets_nents_host[comm_rank + 1] - start;
    Omega_h::Write<LO> elem_sbars = picparts.createCommArray<LO>(dim, 1, 0);
    numberCore(start, nce, device_map, local_sbar, elem_sbars);

    //Communicate core numbering to buffers
    int nbuffers = picparts.numBuffers(dim) - 1;
    Omega_h::HostWrite<LO> buffer_ranks = picparts.bufferedRanks(dim);
    MPI_Request* send_requests = new MPI_Request[nbuffers];
    MPI_Request* recv_requests = new MPI_Request[nbuffers];
    for (int i = 0; i < nbuffers; ++i) {
      int buffer_rank = buffer_ranks[i];
      int nents_start = offsets_nents_host[buffer_rank];
      int nents_size = offsets_nents_host[buffer_rank + 1] - nents_start;
      PS_Comm_Irecv(elem_sbars.view(), nents_start, nents_size,
                    buffer_rank, 0, comm->get_impl(), recv_requests + i);
      PS_Comm_Isend(elem_sbars.view(), start, nce,
                    buffer_rank, 0, comm->get_impl(), send_requests + i);
    }

    PS_Comm_Waitall<Kokkos::DefaultExecutionSpace>(nbuffers, recv_requests,
                                                   MPI_STATUSES_IGNORE);
    delete [] recv_requests;
    PS_Comm_Waitall<Kokkos::DefaultExecutionSpace>(nbuffers, send_requests,
                                                   MPI_STATUSES_IGNORE);
    delete [] send_requests;

    //Convert comm array indices to mesh indices
    Omega_h::LOs comm_indices = picparts.commArrayIndex(dim);
    Omega_h::LOs elem_sbars_tag = convertToMeshIndices(elem_sbars, comm_indices);
    picparts->add_tag(dim, "sbar_id", 1, elem_sbars_tag);

    //Convert sbar_ids to correct values
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); ++itr) {
      itr->second = map[itr->second];
    }
  }

  void ParticleBalancer::buildNgraph(Omega_h::CommPtr comm) {
    int comm_rank = comm->rank();
    weightGraph = agi::createEmptyGraph();

    //Build vert ids and count number of pins
    std::unordered_map<int, agi::lid_t> sbar_to_vert_host;
    agi::gid_t* verts = new agi::gid_t[sbar_ids.size() + 1];
    int index = 0;
    int npins = 0;
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); ++itr, ++index) {
      npins += itr->first.size();
      int i = 0;
      for (auto pitr = itr->first.begin(); pitr != itr->first.end(); ++pitr, ++i) {
        if (*pitr == comm_rank) {
          verts[index] = itr->second + i;
          vert_to_sbar[itr->second + i] = itr->second;
          sbar_to_vert_host[itr->second] = index;
          break;
        }
      }
    }
    verts[index] = comm_rank + max_sbar + 1;
    weightGraph->constructVerts(true, sbar_ids.size() + 1, verts);
    delete [] verts;

    sbar_to_vert = Kokkos::UnorderedMap<int, agi::lid_t>(sbar_to_vert_host.size());
    buildMap(sbar_to_vert_host, sbar_to_vert);
    //Build edge numbering and pins
    agi::gid_t* edges = new agi::gid_t[sbar_ids.size()];
    agi::lid_t* degs = new agi::lid_t[sbar_ids.size()];
    agi::gid_t* pins = new agi::gid_t[npins];
    index = 0;
    int pin_index = 0;
    for (auto itr = sbar_ids.begin(); itr != sbar_ids.end(); ++itr, ++index) {
      edges[index] = itr->second;
      degs[index] = itr->first.size();
      int i = 0;
      for (auto pitr = itr->first.begin(); pitr != itr->first.end(); ++pitr, ++i) {
        pins[pin_index++] = itr->second + i;
        if (*pitr != comm_rank)
          vert_to_owner[itr->second + i] = *pitr;
      }
    }
    weightGraph->constructEdges(sbar_ids.size(), edges, degs, pins);
    weightGraph->constructGhosts(vert_to_owner);
    delete [] pins;
    delete [] degs;
    delete [] edges;

    agi::gid_t global_vtx = weightGraph->numGlobalVtxs();
    agi::gid_t global_edges = weightGraph->numGlobalEdges();
    agi::gid_t global_pins = weightGraph->numGlobalPins();
    if (!comm_rank) {
      pPrintInfo("Ngraph global stats <vtx edges pins>: %ld %ld %ld\n",
             global_vtx,  global_edges, global_pins);
    }
  }

  Omega_h::LOs ParticleBalancer::getSbarIDs(Mesh& picparts) const {
    return picparts->get_array<LO>(picparts->dim(), "sbar_id");
  }

  ParticlePlan ParticleBalancer::balance(double tol, double step_factor) {
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size == 1)
      return ParticlePlan();
    engpar::WeightInput* input = engpar::createWeightInput(weightGraph, tol, step_factor, 0);
    engpar::balanceWeights(input, 0);
    agi::WeightPartitionMap* ptn = weightGraph->getWeightPartition();

    int num_sbars = 0;
    int num_indices = 0;
    for (auto itr = ptn->begin(); itr != ptn->end(); ++itr) {
      ++num_sbars;
      num_indices += itr->second.size() + 1;
    }
    std::unordered_map<LO, LO> sbar_index_map;
    Omega_h::HostWrite<LO> tgt_parts_host(num_indices, "tgt_parts_host");
    Omega_h::HostWrite<Omega_h::Real> wgts_host(num_indices, "wgts_host");
    int tgt_index = 0;
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    for (auto itr = ptn->begin(); itr != ptn->end(); ++itr) {
      int sbar = vert_to_sbar[itr->first];
      sbar_index_map[sbar] = tgt_index;
      for (auto migr_itr = itr->second.begin(); migr_itr != itr->second.end();
           ++migr_itr, ++tgt_index) {
        tgt_parts_host[tgt_index] = vert_to_owner[migr_itr->first];
        wgts_host[tgt_index] = migr_itr->second;
      }
      tgt_parts_host[tgt_index] = -1;
      wgts_host[tgt_index++] = 0;
    }
    return ParticlePlan(sbar_index_map, Omega_h::Write<LO>(tgt_parts_host),
                        Omega_h::Write<Omega_h::Real>(wgts_host));
  }

  ParticlePlan::ParticlePlan(){

  }
  ParticlePlan::ParticlePlan(std::unordered_map<LO, LO>& sbar_index_map,
                             Omega_h::Write<LO> tgt_parts, Omega_h::Write<Omega_h::Real> wgts)
    : sbar_to_index(sbar_index_map.size()), part_ids(tgt_parts), send_wgts(wgts){
    buildMap(sbar_index_map, sbar_to_index);
  }
}
