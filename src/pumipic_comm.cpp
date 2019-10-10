#include "pumipic_mesh.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_array_ops.hpp>
#include <mpi.h>
#include <Omega_h_comm.hpp>

using Omega_h::MpiTraits;

namespace pumipic {
  void Mesh::setupComm(int edim, Omega_h::LOs global_ents_per_rank,
                       Omega_h::LOs picpart_ents_per_rank,
                       Omega_h::LOs ent_owners) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int nents = picpart->nents(edim); 
    Omega_h::Write<Omega_h::LO> ent_rank_lids(nents,0);
    Omega_h::Write<Omega_h::LO> comm_arr_index(nents,0);

    //Count the number of parts if it hasn't been set yet
    if (num_cores[edim] == 0) {
      Omega_h::Write<Omega_h::LO> has_part(comm_size, 0);
      auto markCores = OMEGA_H_LAMBDA(const Omega_h::LO& ent_id) {
        const Omega_h::LO owner = ent_owners[ent_id];
        has_part[owner] = 1;
      };
      Omega_h::parallel_for(nents, markCores, "markCores");
      num_cores[edim] = Omega_h::get_sum(Omega_h::Read<Omega_h::LO>(has_part)) - 1;
    }

    //Compute the number of parts that make up the picpart
    Omega_h::HostRead<Omega_h::LO> host_offset_nents(picpart_ents_per_rank);
    buffered_parts[edim] = Omega_h::HostWrite<Omega_h::LO>(num_cores[edim]);
    int index = 0;
    for (int i = 0; i < host_offset_nents.size()-1; ++i) {
      if (host_offset_nents[i] != host_offset_nents[i+1] && i != commptr->rank())
        buffered_parts[edim][index++] = i;
    }

    //Calculate rankwise local ids
    //First number all entities by the global numbering
    //  NOTE: This numbering is wrong for boundary part ents
    auto global_ids = global_ids_per_dim[edim];
    auto calculateRankLids = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner = ent_owners[ent_id];
      ent_rank_lids[ent_id] = global_ids[ent_id] - global_ents_per_rank[owner];
    };
    Omega_h::parallel_for(nents, calculateRankLids, "calculateRankLids");

    Omega_h::Write<Omega_h::LO> is_complete(comm_size,0);
    Omega_h::Write<Omega_h::LO> boundary_degree(comm_size,0);
    //Mark parts that are completly buffered
    auto checkCompleteness = OMEGA_H_LAMBDA(const Omega_h::LO part_id) {
      const Omega_h::LO global_diff = global_ents_per_rank[part_id+1] -
        global_ents_per_rank[part_id];
      const Omega_h::LO picpart_diff =
      picpart_ents_per_rank[part_id+1] - picpart_ents_per_rank[part_id];
      is_complete[part_id] = (global_diff == picpart_diff) + (picpart_diff != 0);
      boundary_degree[part_id] = picpart_diff * (is_complete[part_id] == 1);
    };
    Omega_h::parallel_for(comm_size, checkCompleteness, "checkCompleteness");
      
    //Renumber the boundary part ents using atomics
    //  NOTE: We can do this for boundary ents because the order doesn't need to be consistent
    Omega_h::Write<Omega_h::LO> picpart_offsets_tmp(nents, 0, "picpart_offsets_tmp");
    auto renumberBoundaryLids = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner = ent_owners[ent_id];
      if (is_complete[owner] == 1) {
        ent_rank_lids[ent_id] = Kokkos::atomic_fetch_add(&(picpart_offsets_tmp[owner]),1);
      }
    };
    Omega_h::parallel_for(nents, renumberBoundaryLids, "renumberBoundaryLids");

    
    //Calculate communication array indices
    // Index = rank_lid + picpart_ents_per_rank
    auto calculateCommArrayIndices = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner = ent_owners[ent_id];
      comm_arr_index[ent_id] = ent_rank_lids[ent_id] + picpart_ents_per_rank[owner];
    };
    Omega_h::parallel_for(nents, calculateCommArrayIndices, "calculateCommArrayIndices");

    offset_ents_per_rank_per_dim[edim] = Omega_h::LOs(picpart_ents_per_rank);
    ent_to_comm_arr_index_per_dim[edim] = Omega_h::LOs(comm_arr_index);
    ent_local_rank_id_per_dim[edim] = Omega_h::LOs(ent_rank_lids);
    is_complete_part[edim] = Omega_h::HostRead<Omega_h::LO>(is_complete);
    num_boundaries[edim] = 0;
    num_bounds[edim] = 0;
    if (edim == dim()) {
      return;
    }
    //*************Determine boundary part information**************//

    //Count offset of boundary entities
    Omega_h::LOs boundary_ent_offsets = Omega_h::offset_scan(Omega_h::LOs(boundary_degree));

    Omega_h::LO num_bounded = num_cores[edim] - num_cores[dim()];
    num_bounds[edim] = num_bounded;
    //Alltoall the number of boundary entities to each owner
    Omega_h::HostRead<Omega_h::LO> is_complete_host = is_complete_part[edim];
    Omega_h::HostRead<Omega_h::LO> boundary_degree_host(boundary_degree);
    Omega_h::HostWrite<Omega_h::LO> recv_boundary_degree_host(comm_size);
    MPI_Request alltoall_request;
    MPI_Ialltoall(boundary_degree_host.data(), 1, MPI_INT,
                  recv_boundary_degree_host.data(), 1, MPI_INT,
                  MPI_COMM_WORLD, &alltoall_request);

    //Create buffers to collect boundary entity ids
    Omega_h::HostRead<Omega_h::LO> boundary_ent_offsets_host(boundary_ent_offsets);
    Omega_h::LO num_bound_ents = boundary_ent_offsets_host[comm_size];
    Omega_h::Write<Omega_h::LO> boundary_rlids(num_bound_ents);
    Omega_h::LOs ent_rlids = rank_lids_per_dim[edim];
    auto gatherBoundedEnts = OMEGA_H_LAMBDA(const Omega_h::LO& ent_id) {
      const Omega_h::LO own = ent_owners[ent_id];
      const Omega_h::LO lid = ent_rank_lids[ent_id];
      const Omega_h::LO start_id = boundary_ent_offsets[own];
      if (is_complete[own] == 1) {
        boundary_rlids[start_id + lid] = ent_rlids[ent_id];
      }
    };
    Omega_h::parallel_for(nents,gatherBoundedEnts, "gatherBoundedEnts");

    //Wait for number sends & receives to finish
    MPI_Wait(&alltoall_request, MPI_STATUS_IGNORE);

    //Create offset sum of recv boundary ents
    int num_recv_bounded = 0;
    Omega_h::HostWrite<Omega_h::LO> recv_boundary_offset_host(comm_size + 1);
    recv_boundary_offset_host[0] = 0;
    for (int i = 0; i < comm_size; ++i) {
      recv_boundary_offset_host[i+1] = recv_boundary_offset_host[i]
                                       + recv_boundary_degree_host[i];
      num_recv_bounded += recv_boundary_degree_host[i] > 0;
    }
    num_boundaries[edim] = num_recv_bounded;
    //Compute the parts that have boundaries of this part
    Omega_h::HostWrite<Omega_h::LO> boundary_part_list(num_boundaries[edim]);
    index = 0;
    for (int i = 0; i < recv_boundary_degree_host.size(); ++i) {
      if (recv_boundary_offset_host[i] != recv_boundary_offset_host[i+1] && i != commptr->rank())
        boundary_part_list[index++] = i;
    }
    boundary_parts[edim] = Omega_h::HostWrite<Omega_h::LO>(boundary_part_list);
    int num_recv_bound_ents = recv_boundary_offset_host[comm_size];

    //Send and Recv boundary rank lids
    MPI_Request* send_requests = NULL;
    if (num_bounded > 0)
      send_requests = new MPI_Request[num_bounded];
    MPI_Request* recv_requests = NULL;
    if (num_recv_bounded)
      recv_requests = new MPI_Request[num_recv_bounded];
    index = 0;
    int index2 = 0;
    Omega_h::HostWrite<Omega_h::LO> recv_rlids(num_recv_bound_ents);
    Omega_h::HostWrite<Omega_h::LO> boundary_rlids_host(boundary_rlids);
    for (int i = 0; i < comm_size; ++i) {
      if (is_complete_host[i] == 1) {
        int start = boundary_ent_offsets_host[i];
        int deg = boundary_degree_host[i];
        MPI_Isend(&(boundary_rlids_host[start]), deg, MPI_INT, i, 0,
                  MPI_COMM_WORLD, send_requests + index++);
      }
      if (recv_boundary_degree_host[i] > 0) {
        int start = recv_boundary_offset_host[i];
        int deg = recv_boundary_degree_host[i];
        MPI_Irecv(&(recv_rlids[start]), deg, MPI_INT, i, 0,
                  MPI_COMM_WORLD, recv_requests + index2++);
      }
    }
    if (num_bounded > 0) {
      MPI_Waitall(num_bounded, send_requests, MPI_STATUSES_IGNORE);
      delete [] send_requests;
    }
    if (num_recv_bounded > 0) {
      MPI_Waitall(num_recv_bounded, recv_requests, MPI_STATUSES_IGNORE);
      delete [] recv_requests;
    }
    Omega_h::Write<Omega_h::LO> recv_boundary(recv_rlids);
    offset_bounded_per_dim[edim] =
      Omega_h::HostWrite<Omega_h::LO>(recv_boundary_offset_host);
    bounded_ent_ids[edim] = Omega_h::LOs(recv_boundary);
  }

  template <class T>
  typename Omega_h::Write<T> Mesh::createCommArray(int edim, int num_entries_per_entity,
                                                   T default_value) {
    int nents = picpart->nents(edim);
    int size = num_entries_per_entity * nents;
    return Omega_h::Write<T>(size,default_value);
  }


  //Max and min operations taken from: https://www.geeksforgeeks.org/compute-the-minimum-or-maximum-max-of-two-integers-without-branching/
  //NOTE These only work for ints, not using them for now
  /*
  template <class T>
  OMEGA_H_INLINE T maxReduce(T x, T y) {
    return x ^ ((x ^ y) & -(x < y));
  }
  template <class T>
  OMEGA_H_INLINE T minReduce(T x, T y) {
    return y ^ ((x ^ y) & -(x < y));
  }
  */

  template <class T>
  OMEGA_H_INLINE T maxReduce(T x, T y) {
    if (x > y)
      return x;
    return y;
  }
  template <class T>
  OMEGA_H_INLINE T minReduce(T x, T y) {
    if (x < y)
      return x;
    return y;
  }

  //Reductions are done by a bulk fan-in fan-out through the core region of each picpart
  template <class T>
  void Mesh::reduceCommArray(int edim, Op op, Omega_h::Write<T> comm_array) {
    int length = comm_array.size();
    int ne = nents(edim);
    int nvals = length / ne;
    if (ne*nvals != length) {
      fprintf(stderr, "Comm array size does not match the expected size for dimension %d\n",edim);
      return;
    }

    //If full mesh then perform an allreduce on the array
    if (isFullMesh()) {
      Omega_h::HostWrite<T> array_host(comm_array);
      MPI_Op mpi_op;
      if (op == SUM_OP)
        mpi_op = MPI_SUM;
      else if (op == MAX_OP)
        mpi_op = MPI_MAX;
      else if (op == MIN_OP)
        mpi_op = MPI_MIN;
      MPI_Allreduce(MPI_IN_PLACE, array_host.data(), array_host.size(),
                    MpiTraits<T>::datatype(), mpi_op, commptr->get_impl());
      comm_array = Omega_h::Write<T>(array_host);
      return;
    }

    
    //Shift comm_array indexing to bulk communication ordering
    Omega_h::Read<Omega_h::LO> arr_index = commArrayIndex(edim);
    Omega_h::Write<T> array(length, 0);
    auto convertToComm = OMEGA_H_LAMBDA(const Omega_h::LO id) {
      for (int i = 0; i < nvals; ++i) {
        const Omega_h::LO index = arr_index[id];
        array[index*nvals + i] = comm_array[id*nvals + i];
      }
    };
    Omega_h::parallel_for(ne, convertToComm, "convertToComm");

    /***************** Fan In ******************/
    //Move values to host
    Omega_h::HostWrite<T> host_array(array);
    T* data = host_array.data();
    
    //Prepare sending and receiving data of cores to the owner of that region
    Omega_h::HostRead<Omega_h::LO> ent_offsets(offset_ents_per_rank_per_dim[edim]);
    int my_num_entries = ent_offsets[commptr->rank()+1] - ent_offsets[commptr->rank()];
    int num_recvs = num_cores[edim] - num_bounds[edim] + num_boundaries[edim];
    int num_sends = num_cores[edim];
    Omega_h::HostWrite<T>** neighbor_arrays = new Omega_h::HostWrite<T>*[num_recvs];
    for (int i = 0; i < num_recvs; ++i)
      neighbor_arrays[i] = new Omega_h::HostWrite<T>(my_num_entries*nvals);
    MPI_Request* send_requests = new MPI_Request[num_sends];
    MPI_Request* recv_requests = new MPI_Request[num_recvs];
    int index = 0;
    for (int i = 0; i < num_cores[edim]; ++i) {
      int rank = buffered_parts[edim][i];
      int num_entries = ent_offsets[rank+1] - ent_offsets[rank];
      if (num_entries > 0) {
        MPI_Isend(data + ent_offsets[rank]*nvals, num_entries*nvals, MpiTraits<T>::datatype(),
                  rank, is_complete_part[edim][rank], commptr->get_impl(), send_requests + i);
        if (is_complete_part[edim][rank] == 2) {
          T* neighbor_data = neighbor_arrays[index]->data();
          MPI_Irecv(neighbor_data, my_num_entries*nvals, MpiTraits<T>::datatype(), rank, 2,
                    commptr->get_impl(), recv_requests + index++);
        }
      }
    }

    //Recv data from bounding parts
    for (Omega_h::LO i = 0; i < num_boundaries[edim]; ++i) {
      int rank = boundary_parts[edim][i];
      T* neighbor_data = neighbor_arrays[index]->data();
      int size = offset_bounded_per_dim[edim][rank+1] - offset_bounded_per_dim[edim][rank];
      MPI_Irecv(neighbor_data, size*nvals, MpiTraits<T>::datatype(), rank, 1,
                commptr->get_impl(), recv_requests + index++);
    }
    //Wait for recv completion
    Omega_h::LOs bounded_ent_ids_local = bounded_ent_ids[edim];
    for (Omega_h::LO i = 0; i < num_recvs; ++i) {
      int finished_neighbor = -1;
      MPI_Status status;
      MPI_Waitany(num_recvs, recv_requests, &finished_neighbor, &status);
      //When recv finishes copy data to the device and perform op
      const Omega_h::LO start_index = ent_offsets[commptr->rank()]*nvals;
      Omega_h::Write<T> recv_array(*(neighbor_arrays[finished_neighbor]));
      if (status.MPI_TAG == 2) {
        if (op == SUM_OP) {
          auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
            Kokkos::atomic_fetch_add(&(array[start_index + i]),recv_array[i]);
          };
          Omega_h::parallel_for(recv_array.size(), reduce_op, "reduce_op");
        }
        else if (op == MAX_OP) {
          auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
            const T x = array[start_index + i];
            const T y = recv_array[i];
            array[start_index + i] = maxReduce(x,y);
          };
          Omega_h::parallel_for(recv_array.size(), reduce_op, "reduce_op");
        }
        else if (op == MIN_OP) {
          auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
            const T x = array[start_index + i];
            const T y = recv_array[i];
            array[start_index + i] = minReduce(x,y);
          };
          Omega_h::parallel_for(recv_array.size(), reduce_op, "reduce_op");
        }
      }
      else {
        const int size = offset_bounded_per_dim[edim][finished_neighbor+1] -
          offset_bounded_per_dim[edim][finished_neighbor];
        const int start = offset_bounded_per_dim[edim][finished_neighbor];
        if (op == SUM_OP) {
          auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
            int index = bounded_ent_ids_local[start+i];
            for (int j = 0; j < nvals; ++j) {
              Kokkos::atomic_fetch_add(&(array[start_index + index*nvals + j]),
                                       recv_array[i*nvals + j]);
            }
          };
          Omega_h::parallel_for(size, reduce_op, "reduce_op");
        }
        else if (op == MAX_OP) {
          auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
            int index = bounded_ent_ids_local[start+i];
            for (int j = 0; j < nvals; ++j) {
              const Omega_h::LO k = start_index + index*nvals + j;
              const T x = array[k];
              const T y = recv_array[i*nvals + j];
              array[k] = maxReduce(x,y);
            }
          };
          Omega_h::parallel_for(size, reduce_op, "reduce_op");
        }
        else if (op == MIN_OP) {
          auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
            int index = bounded_ent_ids_local[start+i];
            for (int j = 0; j < nvals; ++j) {
              const Omega_h::LO k = start_index + index*nvals + j;
              const T x = array[k];
              const T y = recv_array[i*nvals + j];
              array[k] = minReduce(x,y);
            }
          };
          Omega_h::parallel_for(size, reduce_op, "reduce_op");
        }
      }
      delete neighbor_arrays[finished_neighbor];
    }
    MPI_Waitall(num_sends, send_requests,MPI_STATUSES_IGNORE);
    delete [] neighbor_arrays;
    
    /***************** Fan Out ******************/
    //Flip the sizes of the request arrays
    MPI_Request* temp = send_requests;
    send_requests = recv_requests;
    recv_requests = temp;
    int tempi = num_sends;
    num_sends = num_recvs;
    num_recvs = tempi;
    Omega_h::HostWrite<T> reduced_host_array(array);
    data = reduced_host_array.data();
    index = 0;
    for (int i = 0; i < num_cores[edim]; ++i) {
      int rank = buffered_parts[edim][i];
      int num_entries = ent_offsets[rank+1] - ent_offsets[rank];
      if (num_entries > 0) {
        if (is_complete_part[edim][rank]==2) {
          MPI_Isend(data + ent_offsets[commptr->rank()]*nvals, my_num_entries*nvals,
                    MpiTraits<T>::datatype(), rank, 3,
                    commptr->get_impl(), send_requests + index++);
        }
        MPI_Irecv(data + ent_offsets[rank]*nvals, num_entries*nvals, MpiTraits<T>::datatype(),
                  rank, 3, commptr->get_impl(), recv_requests + i);
      }
    }
    //Gather the boundary data to send
    Omega_h::Write<T> boundary_array(bounded_ent_ids_local.size()*nvals);
    const Omega_h::LO start_index = ent_offsets[commptr->rank()]*nvals;
    auto gatherBoundaryData = OMEGA_H_LAMBDA(const Omega_h::LO id) {
      const Omega_h::LO index = bounded_ent_ids_local[id];
      for (int i = 0; i < nvals; ++i)
        boundary_array[id*nvals + i] = array[start_index + index*nvals + i];
    };
    Omega_h::parallel_for(bounded_ent_ids_local.size(),gatherBoundaryData, "gatherBoundaryData");
    
    Omega_h::HostWrite<T> boundary_array_host(boundary_array);
    T* sending_data = boundary_array_host.data();
    for (int i = 0; i < num_boundaries[edim]; ++i) {
      int rank = boundary_parts[edim][i];
      int size = offset_bounded_per_dim[edim][rank+1] - offset_bounded_per_dim[edim][rank];
      int start = offset_bounded_per_dim[edim][rank]*nvals;
      MPI_Isend(sending_data+start, size*nvals, MpiTraits<T>::datatype(), rank, 3,
                commptr->get_impl(), send_requests + index++);
    }
    MPI_Waitall(num_recvs, recv_requests,MPI_STATUSES_IGNORE);
    MPI_Waitall(num_sends, send_requests,MPI_STATUSES_IGNORE);
    delete [] send_requests;
    delete [] recv_requests;

    //Copy reduced array from host to device
    Omega_h::Write<T> reduced_array(reduced_host_array);
    auto setArrayValues = OMEGA_H_LAMBDA(Omega_h::LO i) {
      array[i] = reduced_array[i];
    };
    Omega_h::parallel_for(array.size(), setArrayValues, "setArrayValues");

    auto convertFromComm = OMEGA_H_LAMBDA(const Omega_h::LO id) {
      const Omega_h::LO index = arr_index[id];
      for (int i = 0; i < nvals; ++i)
        comm_array[id*nvals + i] = array[index*nvals + i];
    };
    Omega_h::parallel_for(ne, convertFromComm, "convertFromComm");
  }


#define INST(T)                                                         \
  template Omega_h::Write<T> Mesh::createCommArray(int, int, T);        \
  template void Mesh::reduceCommArray(int, Op, Omega_h::Write<T>);
  
  INST(Omega_h::LO)
  INST(Omega_h::Real)
#undef INST
}
