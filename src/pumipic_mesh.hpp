#pragma once
#include <Omega_h_mesh.hpp>
#include "pumipic_library.hpp"

namespace pumipic {
  class Mesh {
  public:
    //Delete default compilers
    Mesh() = delete; 
    Mesh(const Mesh&) = delete;
      Mesh& operator=(const Mesh&) = delete;

    //Constucts PIC parts with a core and the entire mesh as buffer/safe
    Mesh(Omega_h::Mesh& full_mesh, Omega_h::Write<Omega_h::LO> partition_vector);
    //Constructs PIC parts with a core and buffer all parts within buffer_layers
    // All elements in the core and elements within safe_layers from the core are safe
    Mesh(Omega_h::Mesh& full_mesh, Omega_h::Write<Omega_h::LO> partition_vector,
         int buffer_layers, int safe_layers);
    //TODO? create XGC classification method for creating picpart
    //TODO? create picpart with unsafe_layers instead of safe_layers
    ~Mesh();

    //Returns a pointer to the underlying omega_h mesh
    Omega_h::Mesh* mesh() const {return picpart;}
    //Returns true if the full mesh is buffered
    bool isFullMesh() const;
    //Returns the dimension of the mesh
    int dim() const {return picpart->dim();}
    //Returns the commptr
    Omega_h::CommPtr comm() const {return commptr;}

    //Returns the number of parts buffered
    int numBuffers(int dim) const {return num_cores[dim] + 1;}
    //Returns a host array of the ranks buffered
    Omega_h::HostWrite<Omega_h::LO> bufferedRanks(int dim) const {return buffered_parts[dim];}


    //Picpart global ID array over entities sized nents
    Omega_h::Read<Omega_h::GO> globalIds(int dim) {return global_ids_per_dim[dim];}
    //Safe tag over elements sized nelems (1 - safe, 0 - unsafe)
    Omega_h::Read<Omega_h::LO> safeTag() {return is_ent_safe;}
    //Offset array for number of entities per rank sized comm_size
    Omega_h::Read<Omega_h::LO> nentsOffsets(int dim) {return offset_ents_per_rank_per_dim[dim];}
    //Mapping from local id to comm array index sized nents
    Omega_h::Read<Omega_h::LO> commArrayIndex(int dim) {return ent_to_comm_arr_index_per_dim[dim];}
    //Array of owners of an entity sized nents
    Omega_h::Read<Omega_h::LO> entOwners(int dim) {return ent_owner_per_dim[dim];}
    //The local index of an entity in its own core region sized nents
    Omega_h::Read<Omega_h::LO> rankLocalIndex(int dim) {return ent_local_rank_id_per_dim[dim];}

    //Creates an array of size num_entreis_per_entity * nents for communication
    template <class T>
    typename Omega_h::Write<T> createCommArray(int dim, int num_entries_per_entity,
                                               T default_value);
    enum Op {
      SUM_OP,
      MAX_OP,
      MIN_OP
    };
    //Performs an MPI reduction on a communication array across all picparts
    template <class T>
    void reduceCommArray(int dim, Op op, Omega_h::Write<T> array);

    //Users should not run the following functions. 
    //They are meant to be private, but must be public for enclosing lambdas
    //Picpart construction
    void constructPICPart(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner,
                          Omega_h::Write<Omega_h::GO> elem_gid,
                          Omega_h::LOs rank_offset_nelms,
                          Omega_h::Write<Omega_h::LO> has_part,
                          Omega_h::Write<Omega_h::LO> is_safe);

    //Communication setup
    void setupComm(int dim, Omega_h::LOs global_ents_per_rank,
                   Omega_h::LOs picpart_ents_per_rank,
                   Omega_h::Write<Omega_h::LO> ent_owners);

  private:
    Omega_h::CommPtr commptr;
    Omega_h::Mesh* picpart;

    //PICpart information
    int num_cores[4]; //number of core parts that make up the picpart (The element dimension is smaller than others because of copied entities on the "part" boundary) (doesnt include self)
    Omega_h::HostWrite<Omega_h::LO> buffered_parts[4]; // doesnt include self
    Omega_h::GOs global_ids_per_dim[4]; //Global id of each element
    Omega_h::LOs is_ent_safe; //Safety tag on each element

    //Per Dimension communication information
    Omega_h::LO num_ents_per_dim[4];
    Omega_h::LOs offset_ents_per_rank_per_dim[4]; //An exclusive sum of ents per rank in the picpart
    Omega_h::LOs ent_to_comm_arr_index_per_dim[4]; //mapping from ent to comm arr index
    Omega_h::LOs ent_owner_per_dim[4]; //owning rank of each entity
    Omega_h::LOs ent_local_rank_id_per_dim[4]; //mapping from ent to local index of the core it belongs to.

  };
}
