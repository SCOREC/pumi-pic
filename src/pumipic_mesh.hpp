#pragma once
#include <Omega_h_mesh.hpp>
#include <Omega_h_array.hpp>
#include <stdexcept>
namespace pumipic {
  class Mesh {
  public:
    Mesh(Omega_h::Mesh& full_mesh, Omega_h::Write<Omega_h::LO>& partition_vector,
         int buffer_layers, int safe_layers, int debug = 0);
    //TODO? create XGC classification method for creating picpart
    //TODO? create picpart with the entire mesh
    //TODO? create picpart with unsafe_layers instead of safe_layers
    ~Mesh();
    
    Omega_h::Mesh* mesh() const {return picpart;}
    int num_buffers() const {return num_cores[picpart->dim()] + 1;}
    template <class T>
    typename Omega_h::Write<T> createCommArray(int dim, int num_entries_per_entity,
                                               T default_value);
    enum Op {
      SUM_OP,
      MAX_OP,
      MIN_OP
    };
    template <class T>
    void reduceCommArray(int dim, Op op, Omega_h::Write<T> array);

    
    //private:
    Omega_h::Mesh* picpart;

    //PICpart information
    int num_cores[4]; //number of core parts that make up the picpart (The element dimension is smaller than others because of copied entities on the "part" boundary) (doesnt include self)
    int* buffered_parts[4]; // doesnt include self
    Omega_h::Read<Omega_h::GO> global_ids_per_dim[4]; //Global id of each element
    Omega_h::Read<Omega_h::LO> is_ent_safe; //Safety tag on each element

    //Per Dimension communication information
    Omega_h::Read<Omega_h::LO> offset_ents_per_rank_per_dim[4]; //An exclusive sum of ents per rank in the picpart
    Omega_h::Read<Omega_h::LO> ent_to_comm_arr_index_per_dim[4]; //mapping from ent to comm arr index
    Omega_h::Read<Omega_h::LO> ent_owner_per_dim[4]; //owning rank of each entity
    Omega_h::Read<Omega_h::LO> ent_local_rank_id_per_dim[4]; //mapping from ent to local index of the core it belongs to.

    //Restrict default constructors to crash when called.
    Mesh() {throw std::runtime_error("Cannot build empty PIC part mesh.");}
    Mesh(const Mesh&) {throw std::runtime_error("Cannot copy PIC part mesh.");}
    Mesh& operator=(const Mesh&) {throw std::runtime_error("Cannot copy PIC part mesh.");}

    //Communication setup
    void setupComm(int dim, Omega_h::Write<Omega_h::LO>& global_ents_per_rank,
                   Omega_h::Write<Omega_h::LO>& picpart_ents_per_rank,
                   Omega_h::Write<Omega_h::LO>& ent_owners);
  };
}
