#include "pumipic_comm.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_tag.hpp>
#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_reduce.hpp>

namespace {
  struct PIC_Comm {
    Omega_h::Mesh* picpart;
    int comm_size; //The number of MPI ranks
    int comm_rank; //The MPI rank of this part
    int comm_neighbors; //number of ranks that make up the picpart
    //Exclusive sum of entities per part that is in the picpart
    Omega_h::Read<Omega_h::LO>* nents_per_rank_per_dim[4];
    //Mapping from entity index to comm array index
    Omega_h::Read<Omega_h::LO>* ent_to_comm_arr_per_dim[4]; //can be a tag
    //Owner of each entity
    Omega_h::Read<Omega_h::LO>* ent_owner_per_dim[4]; //can be a tag
    //Mapping from picpart entity index to core local index
    Omega_h::Read<Omega_h::LO>* ent_rank_local_id_per_dim[4]; //can be a tag
    PIC_Comm() {
      for (int i = 0; i < 4; ++i) {
        nents_per_rank_per_dim[i] = NULL;
        ent_to_comm_arr_per_dim[i] = NULL;
        ent_owner_per_dim[i] = NULL;
        ent_rank_local_id_per_dim[i] = NULL;
      }
    }
    ~PIC_Comm() {
      for (int i = 0; i < 4; ++i) {
        if (nents_per_rank_per_dim[i]) {
          delete nents_per_rank_per_dim[i];
          delete ent_to_comm_arr_per_dim[i];
          delete ent_owner_per_dim[i];
          delete ent_rank_local_id_per_dim[i];
        }
      }
    }
  };
  
  PIC_Comm pic_comm;
}

namespace pumipic {
  int PIC_Comm_Self() { return pic_comm.comm_rank;}
  int PIC_Comm_Size() { return pic_comm.comm_size;}
  int PIC_Comm_Neighbors() { return pic_comm.comm_neighbors;}
  void setupPICComm(Omega_h::Mesh* picpart, int dim,
                    Omega_h::Write<Omega_h::LO>& global_ents_per_rank,
                    Omega_h::Write<Omega_h::LO>& picpart_ents_per_rank,
                    Omega_h::Write<Omega_h::LO>& ent_global_numbering,
                    Omega_h::Write<Omega_h::LO>& ent_owners) {
    pic_comm.picpart = picpart;
    MPI_Comm_rank(MPI_COMM_WORLD, &(pic_comm.comm_rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(pic_comm.comm_size));
    int nents = ent_owners.size();
    Omega_h::Write<Omega_h::LO> ent_rank_lids(nents,0);
    Omega_h::Write<Omega_h::LO> comm_arr_index(nents,0);

    //Calculate rankwise local ids
    auto calculateRankLids = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner = ent_owners[ent_id];
      ent_rank_lids[ent_id] = ent_global_numbering[ent_id] - global_ents_per_rank[owner];
    };
    Omega_h::parallel_for(nents, calculateRankLids, "calculateRankLids");
    //Calculate communication array indices
    // Index = rank_lid + picpart_ents_per_rank
    auto calculateCommArrayIndices = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner = ent_owners[ent_id];
      comm_arr_index[ent_id] = ent_rank_lids[ent_id] + picpart_ents_per_rank[owner];
    };
    Omega_h::parallel_for(nents, calculateCommArrayIndices, "calculateCommArrayIndices");
    
    pic_comm.nents_per_rank_per_dim[dim] = new Omega_h::Read<Omega_h::LO>(picpart_ents_per_rank);
    pic_comm.ent_to_comm_arr_per_dim[dim] = new Omega_h::Read<Omega_h::LO>(comm_arr_index);
    pic_comm.ent_owner_per_dim[dim] = new Omega_h::Read<Omega_h::LO>(ent_owners);
    pic_comm.ent_rank_local_id_per_dim[dim] = new Omega_h::Read<Omega_h::LO>(ent_rank_lids);


    printf("Rank %d has %d elements\n", pic_comm.comm_rank,
           picpart_ents_per_rank[pic_comm.comm_size]);
  }

  template <class T>
  typename Omega_h::Write<T> createCommArray(int dim, int num_entries_per_entity,
                                             T default_value) {
    if (!pic_comm.nents_per_rank_per_dim[dim]) {
      fprintf(stderr, "Communication was not setup for dimension %d\n",dim);
      return Omega_h::Write<T>(0,0);
    }
    int num_entries = (*(pic_comm.nents_per_rank_per_dim[dim]))[pic_comm.comm_size];
    int size = num_entries_per_entity * num_entries;
    return Omega_h::Write<T>(size,default_value);
  }

  //Reductions are done by a bulk fan-in fan-out through the core region of each picpart
  template <class T>
  void reduce(int dim, Op op, Omega_h::Write<T>& array) {
    if (!pic_comm.nents_per_rank_per_dim[dim]) {
      fprintf(stderr, "Communication was not setup for dimension %d\n",dim);
      return;
    }
    Omega_h::HostRead<Omega_h::LO> ent_offsets(*pic_comm.nents_per_rank_per_dim[dim]);
    if (ent_offsets[pic_comm.comm_size] != array.size()) {
      fprintf(stderr, "Comm array size does not match the expected size for dimension %d\n",dim);
      return;
    }
    /***************** Fan In ******************/
    //Move values to host
    Omega_h::HostWrite<T> host_array(array);
    T* data = host_array.data();
    
    //Prepare sending and receiving data of cores to the owner of that region
    int my_num_entries = ent_offsets[PIC_Comm_Self()+1] - ent_offsets[PIC_Comm_Self()];
    Omega_h::HostWrite<T>** neighbor_arrays = new Omega_h::HostWrite<T>*[PIC_Comm_Neighbors()];
    for (int i = 0; i < PIC_Comm_Neighbors(); ++i)
      neighbor_arrays[i] = new Omega_h::HostWrite<T>(my_num_entries);
    MPI_Request* send_requests = new MPI_Request[PIC_Comm_Neighbors()];
    MPI_Request* recv_requests = new MPI_Request[PIC_Comm_Neighbors()];
    int neighbor_index = 0;
    //TODO compile a list of neighbor ranks on host
    for (int i = 0; i < PIC_Comm_Size(); ++i) {
      int num_entries = ent_offsets[i+1] - ent_offsets[i];
      if (num_entries > 0 && i != PIC_Comm_Self()) {
        //TODO determine the mpi datatype based on T or use MPI_CHAR and sizeof(T)
        MPI_Isend(data + ent_offsets[i], num_entries, MPI_INT, i, 0,
                  MPI_COMM_WORLD, send_requests + neighbor_index);
        
        T* neighbor_data = neighbor_arrays[neighbor_index]->data();
        MPI_Irecv(neighbor_data, my_num_entries, MPI_INT, i, 0,
                  MPI_COMM_WORLD, recv_requests + neighbor_index);
        ++neighbor_index;
      }
    }

    //Wait for recv completion
    for (int i = 0; i < PIC_Comm_Neighbors(); ++i) {
      int finished_neighbor;
      MPI_Waitany(PIC_Comm_Neighbors(), recv_requests, &finished_neighbor, MPI_STATUS_IGNORE);
      //When recv finishes copy data to the device and perform op
      const Omega_h::LO start_index = ent_offsets[PIC_Comm_Self()];
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
    MPI_Waitall(PIC_Comm_Neighbors(), send_requests,MPI_STATUSES_IGNORE);
    delete [] neighbor_arrays;
    
    /***************** Fan Out ******************/
    Omega_h::HostWrite<T> reduced_host_array(array);
    data = reduced_host_array.data();
    neighbor_index = 0;
    //TODO compile a list of neighbor ranks on host
    for (int i = 0; i < PIC_Comm_Size(); ++i) {
      int num_entries = ent_offsets[i+1] - ent_offsets[i];
      if (num_entries > 0 && i != PIC_Comm_Self()) {
        //TODO determine the mpi datatype based on T or use MPI_CHAR and sizeof(T)
        MPI_Isend(data + ent_offsets[PIC_Comm_Self()], my_num_entries, MPI_INT, i, 1,
                  MPI_COMM_WORLD, send_requests + neighbor_index);
        
        MPI_Irecv(data + ent_offsets[i], num_entries, MPI_INT, i, 1,
                  MPI_COMM_WORLD, recv_requests + neighbor_index);
        ++neighbor_index;
      }
    }
    MPI_Waitall(PIC_Comm_Neighbors(), recv_requests,MPI_STATUSES_IGNORE);
    MPI_Waitall(PIC_Comm_Neighbors(), send_requests,MPI_STATUSES_IGNORE);
    delete [] send_requests;
    delete [] recv_requests;

    //Copy reduced array from host to device
    Omega_h::Write<T> reduced_array(reduced_host_array);
    auto setArrayValues = OMEGA_H_LAMBDA(Omega_h::LO i) {
      array[i] = reduced_array[i];
    };
    Omega_h::parallel_for(array.size(), setArrayValues, "setArrayValues");
  }

}
