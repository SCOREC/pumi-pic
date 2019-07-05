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
    Mesh(Omega_h::Mesh& full_mesh, Omega_h::LOs partition_vector);
    //Constructs PIC parts with a core and buffer all parts within buffer_layers
    // All elements in the core and elements within safe_layers from the core are safe
    Mesh(Omega_h::Mesh& full_mesh, Omega_h::LOs partition_vector,
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
    Omega_h::GOs globalIds(int dim) {return global_ids_per_dim[dim];}
    //Safe tag over elements sized nelems (1 - safe, 0 - unsafe)
    Omega_h::LOs safeTag() {return is_ent_safe;}
    //Offset array for number of entities per rank sized comm_size
    Omega_h::LOs nentsOffsets(int dim) {return offset_ents_per_rank_per_dim[dim];}
    //Mapping from local id to comm array index sized nents
    Omega_h::LOs commArrayIndex(int dim) {return ent_to_comm_arr_index_per_dim[dim];}
    //Array of owners of an entity sized nents
    Omega_h::LOs entOwners(int dim) {return ent_owner_per_dim[dim];}
    //The local index of an entity in its own core region sized nents
    Omega_h::LOs rankLocalIndex(int dim) {return ent_local_rank_id_per_dim[dim];}

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
    void constructPICPart(Omega_h::Mesh& mesh, Omega_h::LOs owner[4],
                          Omega_h::GOs ent_gid_per_dim[4],
                          Omega_h::LOs rank_offset_nents_per_dim[4],
                          Omega_h::Write<Omega_h::LO> has_part,
                          Omega_h::Write<Omega_h::LO> is_safe);

    //Communication setup
    void setupComm(int dim, Omega_h::LOs global_ents_per_rank,
                   Omega_h::LOs picpart_ents_per_rank,
                   Omega_h::LOs ent_owners);

  private:
    Omega_h::CommPtr commptr;
    Omega_h::Mesh* picpart;

    //*********************PICpart information**********************/
    //Number of core parts that are buffered (doesn't include self)
    int num_cores[4];
    //Global ID of each mesh entity per dimension
    Omega_h::GOs global_ids_per_dim[4];
    //Safe tag defined on the mesh elements
    Omega_h::LOs is_ent_safe;

    //Per Dimension communication information
    //List of core parts that are buffered (doesn't include self)
    Omega_h::HostWrite<Omega_h::LO> buffered_parts[4];
    //Exclusive sum of number of entites per rank of buffered_parts/boundary_parts
    //   for each entity dimension
    Omega_h::LOs offset_ents_per_rank_per_dim[4];
    //Mapping from entity id to comm array index
    Omega_h::LOs ent_to_comm_arr_index_per_dim[4];
    //The owning part of each entity per dimension
    Omega_h::LOs ent_owner_per_dim[4];
    //Mapping from entity id to local index of the core it belongs to
    //  Note: This is stored in case needed, but is not used beyond setup.
    Omega_h::LOs ent_local_rank_id_per_dim[4];
    //NOTE: Bounding means that the rank 'owns' at least one entity on the picpart boundary of the bounded part
    //List (per dimension) of parts that only the boundary exist in the picpart
    //  Note: boundary_parts[mesh_dim] is empty
    Omega_h::HostWrite<Omega_h::LO> boundary_parts[4];
    //List (per dimension) of parts that have the boundary of this core
    //  Note: bounded_parts[mesh_dim] is empty
    Omega_h::HostWrite<Omega_h::LO> bounded_parts[4];
    //Exclusive sum of number of bounded entities to send to bounded parts
    // size = bounded_parts.size()+1
    Omega_h::LOs offset_bounded_ents_per_rank_per_dim[4];
    //Mapping for each rank this part bounds
    Omega_h::LOs bounded_ents_to_comm_array_per_dim[4];
  };
}
