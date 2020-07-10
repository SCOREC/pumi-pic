#pragma once
#include <Omega_h_mesh.hpp>
#include "pumipic_library.hpp"
#include "pumipic_input.hpp"

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
    //Create picparts from input structure
    Mesh(Input&);
    ~Mesh();

    //Returns true if the full mesh is buffered
    bool isFullMesh() const;
    //Calls function on the omega_h mesh
    Omega_h::Mesh* operator->() {return picpart;}
    //Returns a pointer to the underlying omega_h mesh
    Omega_h::Mesh* mesh() const {return picpart;}
    //Returns the dimension of the mesh
    int dim() const {return picpart->dim();}
    //Returns the number of entities of the picpart
    Omega_h::LO nents(int dim) const {return picpart->nents(dim);}
    //Returns the number of elements of the picpart
    Omega_h::LO nelems() const {return picpart->nelems();}
    //Returns the commptr
    Omega_h::CommPtr comm() const {return commptr;}

    //Returns the number of parts buffered (includes self in count)
    int numBuffers(int dim) const {return num_cores[dim] + 1;}
    //Returns a host array of the ranks buffered (Doesn't include self)
    Omega_h::HostWrite<Omega_h::LO> bufferedRanks(int dim) const {return buffered_parts[dim];}


    //Picpart global ID array over entities sized nents
    Omega_h::GOs globalIds(int dim) {return picpart->get_array<Omega_h::GO>(dim, "gids");}
    //Safe tag over elements sized nelems (1 - safe, 0 - unsafe)
    Omega_h::LOs safeTag() {return picpart->get_array<Omega_h::LO>(dim(), "safe");}
    //Array of owners of an entity sized nents
    Omega_h::LOs entOwners(int dim) {return picpart->get_array<Omega_h::LO>(dim, "ownership");}
    //The local index of an entity in its own core region sized nents
    Omega_h::LOs rankLocalIndex(int dim) {return picpart->get_array<Omega_h::LO>(dim,"rank_lids");}
    //Offset array for number of entities per rank sized comm_size
    Omega_h::LOs nentsOffsets(int dim) {return offset_ents_per_rank_per_dim[dim];}
    //Mapping from local id to comm array index sized nents
    Omega_h::LOs commArrayIndex(int dim) {return ent_to_comm_arr_index_per_dim[dim];}

    //Creates an array of size num_entreis_per_entity * nents for communication
    template <class T>
    typename Omega_h::Write<T> createCommArray(int dim, int num_entries_per_entity,
                                               T default_value);
    enum Op {
      SUM_OP, //Sum contributions
      MAX_OP, //Take max of all contributions
      MIN_OP, //Take min of all contributions
      BCAST_OP //Take the owner's value
    };
    //Performs an MPI reduction on a communication array across all picparts
    template <class T>
    void reduceCommArray(int dim, Op op, Omega_h::Write<T> array);

    //Users should not run the following functions.
    //They are meant to be private, but must be public for enclosing lambdas
    //Picpart construction
    void constructPICPart(Omega_h::Mesh& mesh, Omega_h::CommPtr comm,
                          Omega_h::LOs owner,
                          Omega_h::Write<Omega_h::LO> has_part,
                          Omega_h::Write<Omega_h::LO> is_safe,
                          bool render = false);

    //Communication setup
    void setupComm(int dim, Omega_h::LOs global_ents_per_rank,
                   Omega_h::LOs picpart_ents_per_rank,
                   Omega_h::LOs ent_owners);

  private:
    Omega_h::CommPtr commptr;
    Omega_h::Mesh* picpart;

    bool is_full_mesh;

    //*********************PICpart information**********************/
    //Number of core parts that are buffered (doesn't include self)
    int num_cores[4];
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
    //Mapping from entity id to local index of the core it belongs to
    //  Note: This is stored in case needed, but is not used beyond setup.
    Omega_h::LOs ent_local_rank_id_per_dim[4];
    /*Flag for each buffered part. True if the entire part is buffered
     * 2 = complete
     * 1 = partial
     * 0 = empty
     */
    Omega_h::HostRead<Omega_h::LO> is_complete_part[4];
    //Number of parts this part bounds
    int num_bounds[4];
    //Number of parts that have boundaries of this part
    int num_boundaries[4];
    //List of parts that have boundaries of this part
    Omega_h::HostWrite<Omega_h::LO> boundary_parts[4];
    //Exclusive sum of number of bounded entities to send to bounded parts
    // size = bounded_parts.size()+1
    Omega_h::HostWrite<Omega_h::LO> offset_bounded_per_dim[4];
    //The entities to send to each part for boundary
    Omega_h::LOs bounded_ent_ids[4];
  };
}
