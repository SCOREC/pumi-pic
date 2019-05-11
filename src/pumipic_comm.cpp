#include "pumipic_mesh.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_tag.hpp>
#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_reduce.hpp>

namespace pumipic {
  void Mesh::setupComm(int dim, Omega_h::Write<Omega_h::LO>& global_ents_per_rank,
                    Omega_h::Write<Omega_h::LO>& picpart_ents_per_rank,
                    Omega_h::Write<Omega_h::LO>& ent_owners) {
    Omega_h::CommPtr comm = picpart->comm();
    int nents = picpart->nents(dim);
    Omega_h::Write<Omega_h::LO> ent_rank_lids(nents,0);
    Omega_h::Write<Omega_h::LO> comm_arr_index(nents,0);

    //Compute the number of parts that make up the picpart
    Omega_h::HostWrite<Omega_h::LO> host_offset_nents(picpart_ents_per_rank);
    buffered_parts[dim] = new int[num_cores[dim]];
    int index = 0;
    for (int i = 0; i < host_offset_nents.size(); ++i) {
      if (host_offset_nents[i] != host_offset_nents[i+1] && i != comm->rank())
        buffered_parts[dim][index++] = i;
    }
    
    //Calculate rankwise local ids
    auto calculateRankLids = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner = ent_owners[ent_id];
      ent_rank_lids[ent_id] = global_ids_per_dim[dim][ent_id] - global_ents_per_rank[owner];
    };
    Omega_h::parallel_for(nents, calculateRankLids, "calculateRankLids");
    //Calculate communication array indices
    // Index = rank_lid + picpart_ents_per_rank
    auto calculateCommArrayIndices = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner = ent_owners[ent_id];
      comm_arr_index[ent_id] = ent_rank_lids[ent_id] + picpart_ents_per_rank[owner];
    };
    Omega_h::parallel_for(nents, calculateCommArrayIndices, "calculateCommArrayIndices");
    
    offset_ents_per_rank_per_dim[dim] = Omega_h::Read<Omega_h::LO>(picpart_ents_per_rank);
    ent_to_comm_arr_index_per_dim[dim] = Omega_h::Read<Omega_h::LO>(comm_arr_index);
    ent_owner_per_dim[dim] = Omega_h::Read<Omega_h::LO>(ent_owners);
    ent_local_rank_id_per_dim[dim] = Omega_h::Read<Omega_h::LO>(ent_rank_lids);
  }

  template <class T>
  typename Omega_h::Write<T> Mesh::createCommArray(int dim, int num_entries_per_entity,
                                                   T default_value) {
    Omega_h::CommPtr comm = picpart->comm();
    int num_entries = (offset_ents_per_rank_per_dim[dim])[comm->size()];
    int size = num_entries_per_entity * num_entries;
    return Omega_h::Write<T>(size,default_value);
  }

  //Reductions are done by a bulk fan-in fan-out through the core region of each picpart
  template <class T>
  void Mesh::reduceCommArray(int dim, Op op, Omega_h::Write<T> array) {
    Omega_h::CommPtr comm = picpart->comm();
    Omega_h::HostRead<Omega_h::LO> ent_offsets(offset_ents_per_rank_per_dim[dim]);
    if (ent_offsets[comm->size()] != array.size()) {
      fprintf(stderr, "Comm array size does not match the expected size for dimension %d\n",dim);
      return;
    }
    /***************** Fan In ******************/
    //Move values to host
    Omega_h::HostWrite<T> host_array(array);
    T* data = host_array.data();
    
    //Prepare sending and receiving data of cores to the owner of that region
    int my_num_entries = ent_offsets[comm->rank()+1] - ent_offsets[comm->rank()];
    Omega_h::HostWrite<T>** neighbor_arrays = new Omega_h::HostWrite<T>*[num_cores[dim]];
    for (int i = 0; i < num_cores[dim]; ++i)
      neighbor_arrays[i] = new Omega_h::HostWrite<T>(my_num_entries);
    MPI_Request* send_requests = new MPI_Request[num_cores[dim]];
    MPI_Request* recv_requests = new MPI_Request[num_cores[dim]];
    //TODO compile a list of neighbor ranks on host
    for (int i = 0; i < num_cores[dim]; ++i) {
      int rank = buffered_parts[dim][i];
      int num_entries = ent_offsets[rank+1] - ent_offsets[rank];
      if (num_entries > 0) {
        //TODO determine the mpi datatype based on T or use MPI_CHAR and sizeof(T)
        MPI_Isend(data + ent_offsets[rank], num_entries, MPI_INT, rank, 0,
                  comm->get_impl(), send_requests + i);
        
        T* neighbor_data = neighbor_arrays[i]->data();
        MPI_Irecv(neighbor_data, my_num_entries, MPI_INT, rank, 0,
                  comm->get_impl(), recv_requests + i);
      }
    }

    //Wait for recv completion
    for (int i = 0; i < num_cores[dim]; ++i) {
      int finished_neighbor;
      MPI_Waitany(num_cores[dim], recv_requests, &finished_neighbor, MPI_STATUS_IGNORE);
      //When recv finishes copy data to the device and perform op
      const Omega_h::LO start_index = ent_offsets[comm->rank()];
      Omega_h::Write<T> recv_array(*(neighbor_arrays[finished_neighbor]));
      if (op == SUM_OP) {
        auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
          array[start_index + i] += recv_array[i];
        };
        Omega_h::parallel_for(recv_array.size(), reduce_op, "reduce_op");
      }
      //Max and min operations taken from: https://www.geeksforgeeks.org/compute-the-minimum-or-maximum-max-of-two-integers-without-branching/
      else if (op == MAX_OP) {
        auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
          const Omega_h::LO x = array[start_index + i];
          const Omega_h::LO y = recv_array[i];
          array[start_index + i] = x ^ ((x ^ y) & -(x < y));;
        };
        Omega_h::parallel_for(recv_array.size(), reduce_op, "reduce_op");
      }
      else if (op == MIN_OP) {
        auto reduce_op = OMEGA_H_LAMBDA(Omega_h::LO i) {
          const Omega_h::LO x = array[start_index + i];
          const Omega_h::LO y = recv_array[i];
          array[start_index + i] = y ^ ((x ^ y) & -(x < y));
        };
        Omega_h::parallel_for(recv_array.size(), reduce_op, "reduce_op");
      }
      delete neighbor_arrays[finished_neighbor];
    }
    MPI_Waitall(num_cores[dim], send_requests,MPI_STATUSES_IGNORE);
    delete [] neighbor_arrays;
    
    /***************** Fan Out ******************/
    Omega_h::HostWrite<T> reduced_host_array(array);
    data = reduced_host_array.data();
    //TODO compile a list of neighbor ranks on host
    for (int i = 0; i < num_cores[dim]; ++i) {
      int rank = buffered_parts[dim][i];
      int num_entries = ent_offsets[rank+1] - ent_offsets[rank];
      if (num_entries > 0) {
        //TODO determine the mpi datatype based on T or use MPI_CHAR and sizeof(T) (use mpi traits in omega_h?)
        MPI_Isend(data + ent_offsets[comm->rank()], my_num_entries, MPI_INT, rank, 1,
                  comm->get_impl(), send_requests + i);
        
        MPI_Irecv(data + ent_offsets[rank], num_entries, MPI_INT, rank, 1,
                  comm->get_impl(), recv_requests + i);
      }
    }
    MPI_Waitall(num_cores[dim], recv_requests,MPI_STATUSES_IGNORE);
    MPI_Waitall(num_cores[dim], send_requests,MPI_STATUSES_IGNORE);
    delete [] send_requests;
    delete [] recv_requests;

    //Copy reduced array from host to device
    Omega_h::Write<T> reduced_array(reduced_host_array);
    auto setArrayValues = OMEGA_H_LAMBDA(Omega_h::LO i) {
      array[i] = reduced_array[i];
    };
    Omega_h::parallel_for(array.size(), setArrayValues, "setArrayValues");
  }


#define INST(T)                                                         \
  template Omega_h::Write<T> Mesh::createCommArray(int, int, T);        \
  template void Mesh::reduceCommArray(int, Op, Omega_h::Write<T>);
  
  INST(Omega_h::LO)
  INST(Omega_h::Real)
#undef INST
}
