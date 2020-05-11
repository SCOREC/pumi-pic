#include "pumipic_mesh.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_scan.hpp>
#include <Omega_h_file.hpp>

namespace {
  void setOwnerByClassification(Omega_h::Mesh& m, Omega_h::LOs class_owners, int self,
                                Omega_h::Write<Omega_h::LO> owns);
  Omega_h::LOs defineOwners(Omega_h::Mesh& m, int dim, Omega_h::CommPtr, Omega_h::LOs owner);
  Omega_h::LOs calculateOwnerOffset(Omega_h::LOs owner, int comm_size);
  Omega_h::LOs createGlobalNumbering(Omega_h::LOs owner, int comm_size,
                                     Omega_h::Write<Omega_h::GO> elem_gid);
  Omega_h::LOs rankLidNumbering(Omega_h::LOs owner, Omega_h::LOs offset, Omega_h::GOs gids);
  void bfsBufferLayers(Omega_h::Mesh& mesh, int bridge_dim, Omega_h::CommPtr comm,
                       int safe_layers, int ghost_layers,
                       Omega_h::Write<Omega_h::LO> is_safe,
                       Omega_h::LOs owner, Omega_h::Write<Omega_h::LO> has_part);
  void setSafeEnts(Omega_h::Mesh& mesh, int dim, int size, Omega_h::Write<Omega_h::LO> has_part,
                   Omega_h::LOs owner, Omega_h::Write<Omega_h::LO> buf);
  Omega_h::LO sumPositives(Omega_h::LO size, Omega_h::Write<Omega_h::LO> arr);
  void gatherCoords(Omega_h::Mesh& mesh, Omega_h::LOs vert_ids,
                    Omega_h::Write<Omega_h::Real> new_coords);
  void buildAndClassify(Omega_h::Mesh& full_mesh, Omega_h::Mesh* picpart, int dim, int num_ents,
                        Omega_h::LOs ent_ids, Omega_h::LOs vert_ids,
                        Omega_h::Write<Omega_h::Real> new_coords);
}

namespace pumipic {
  Mesh::Mesh(Omega_h::Mesh& mesh, Omega_h::LOs owner) {
    Omega_h::CommPtr comm = mesh.library()->world();
    int rank = comm->rank();
    int comm_size = comm->size();

    /*********** Set safe zone and buffer to be entire mesh****************/
    Omega_h::Write<Omega_h::LO> is_safe(mesh.nelems(), 1);
    Omega_h::Write<Omega_h::LO> has_part(comm_size, 1);
    is_full_mesh = true;
    constructPICPart(mesh, comm, owner, has_part, is_safe);
  }

  Mesh::Mesh(Omega_h::Mesh& mesh, Omega_h::LOs owner,
                  int ghost_layers, int safe_layers) {
    Omega_h::CommPtr comm = mesh.library()->world();
    int rank = comm->rank();
    int comm_size = comm->size();
    if (ghost_layers < safe_layers) {
      if (!rank)
        fprintf(stderr, "Ghost layers must be >= safe layers");
      throw 1;
    }
    is_full_mesh = false;
    // **********Determine safe zone and ghost region**************** //
    Omega_h::Write<Omega_h::LO> is_safe(mesh.nelems());
    Omega_h::Write<Omega_h::LO> has_part(comm_size);
    int bridge_dim = 0;
    bfsBufferLayers(mesh, bridge_dim, comm, safe_layers, ghost_layers, is_safe,
                    owner, has_part);

    constructPICPart(mesh, comm, owner, has_part, is_safe);
  }

  Mesh::Mesh(Input& in) {
    Omega_h::CommPtr comm = in.comm;
    int rank = comm->rank();
    int comm_size = comm->size();

    /*********** Set safe zone and buffer to be entire mesh****************/
    Omega_h::LOs owners = in.partition;
    if (in.ownership_rule == Input::CLASSIFICATION) {
      Omega_h::Write<Omega_h::LO> owns(in.m.nelems(), "owns_w");
      setOwnerByClassification(in.m, in.partition, rank, owns);
      owners = Omega_h::LOs(owns);
    }
    Omega_h::Write<Omega_h::LO> is_safe(in.m.nelems(),in.safeMethod==Input::FULL, "is_safe");
    Omega_h::Write<Omega_h::LO> has_part(comm_size,1, "has_part");
    if ((in.safeMethod != Input::NONE && in.safeMethod != Input::FULL)
        || in.bufferMethod != Input::FULL) {
      Omega_h::Write<Omega_h::LO> safe(in.m.nelems(), 0, "safe");
      Omega_h::Write<Omega_h::LO> part(comm_size, 0, "part");

      bfsBufferLayers(in.m, in.bridge_dim, comm, in.safeBFSLayers, in.bufferBFSLayers, safe,
                      owners, part);

      if (in.safeMethod == Input::BFS || in.safeMethod == Input::MINIMUM)
        is_safe = safe;
      if (in.bufferMethod == Input::BFS || in.bufferMethod == Input::MINIMUM)
        has_part = part;
    }

    if (in.bufferMethod == Input::FULL)
      is_full_mesh = true;
    else
      is_full_mesh = false;

    constructPICPart(in.m, in.comm, owners, has_part, is_safe);
  }

  void Mesh::constructPICPart(Omega_h::Mesh& mesh, Omega_h::CommPtr comm,
                              Omega_h::LOs owner, Omega_h::Write<Omega_h::LO> has_part,
                              Omega_h::Write<Omega_h::LO> is_safe, bool render) {
    int rank = comm->rank();
    int comm_size = comm->size();
    int dim = mesh.dim();

    /************* Define Ownership of each lower dimension entity ************/
    Omega_h::LOs owner_dim[4];
    for (int i = 0; i < dim; ++i) {
      owner_dim[i] = defineOwners(mesh, i, comm, owner);
      mesh.add_tag(i, "ownership", 1, owner_dim[i]);
    }
    owner_dim[dim] = owner;
    mesh.add_tag(dim, "ownership", 1, owner_dim[dim]);

    if (render && rank == 0)
      Omega_h::vtk::write_parallel("partition", &mesh, dim);
    /************* Globally Number Entities **********/
    Omega_h::GOs ent_gid_per_dim[4];
    Omega_h::LOs rank_lid_per_dim[4];
    Omega_h::LOs rank_offset_nents[4];
    for (int i = 0; i <= dim; ++i) {
      Omega_h::Write<Omega_h::GO> gids(mesh.nents(i), "global_ids");
      rank_offset_nents[i] = createGlobalNumbering(owner_dim[i], comm_size, gids);
      ent_gid_per_dim[i] = Omega_h::GOs(gids);
      rank_lid_per_dim[i] = rankLidNumbering(owner_dim[i], rank_offset_nents[i],
                                             ent_gid_per_dim[i]);
    }

    /***************** Count the number of parts in the picpart ****************/
    num_cores[dim] = sumPositives(has_part.size(),has_part) - 1;
    for (int i = 0; i < dim; ++i)
      num_cores[i] = 0;
    /***************** Count Number of Entities in the PICpart *************/
    //Mark all entities owned by a part with has_part[part] = true as staying
    Omega_h::Write<Omega_h::LO> buf_ents[4];
    for (int i = 0; i <= dim; ++i)
      buf_ents[i] = Omega_h::Write<Omega_h::LO>(mesh.nents(i),0);
    for (int i = 0; i <= dim; ++i)
      setSafeEnts(mesh, i, mesh.nelems(), has_part, owner_dim[dim], buf_ents[i]);

    //Gather number of entities remaining in the picpart
    Omega_h::GO* num_ents = new Omega_h::GO[dim+1];
    for (int i = 0; i <= dim; ++i)
      num_ents[i] = sumPositives(mesh.nents(i), buf_ents[i]);

    /**************** Create numberings for the entities on the picpart **************/
    Omega_h::LOs ent_ids[4];
    for (int i = 0; i <= dim; ++i) {
      //Default the value to the number of entities in the pic part (for padding)
      Omega_h::Write<Omega_h::LO> numbering(mesh.nents(i), -1);
      const auto size = mesh.nents(i);
      auto is_valid = buf_ents[i];
      Omega_h::LOs is_valid_r(is_valid);
      auto offset = Omega_h::offset_scan(is_valid_r);
      Omega_h::parallel_for(size, OMEGA_H_LAMBDA(Omega_h::LO i) {
          if(is_valid_r[i])
            numbering[i] = offset[i];
      });
      ent_ids[i] = numbering;
    }

    //If full mesh buffer then we don't need to make new mesh for the picparts
    if (isFullMesh()) {
      //Set picpart to point to the mesh
      picpart = &mesh;
      is_ent_safe = Omega_h::LOs(is_safe);
      commptr = comm;
      //Save all tags on full mesh
      for (int i = 0; i <= dim; ++i) {
        global_ids_per_dim[i] = ent_gid_per_dim[i];
        rank_lids_per_dim[i] = rank_lid_per_dim[i];
        ent_owner_per_dim[i] = owner_dim[i];
        //We dont need to setup communication arrays because an allreduce is used
        Omega_h::LOs picpart_offset_nents = calculateOwnerOffset(owner_dim[i], comm_size);
        setupComm(i, rank_offset_nents[i], picpart_offset_nents, owner_dim[i]);

      }
      delete [] num_ents;
      return;
    }

    //************Build a new mesh as the picpart**************
    Omega_h::Library* lib = mesh.library();
    picpart = new Omega_h::Mesh(lib);

    //Gather coordinates
    Omega_h::Write<Omega_h::Real> new_coords((num_ents[0])*dim,0);
    gatherCoords(mesh, ent_ids[0], new_coords);

    //Build the mesh
    for (int i = dim; i >= 0; --i)
      buildAndClassify(mesh,picpart,i,num_ents[i], ent_ids[i], ent_ids[0], new_coords);
    Omega_h::finalize_classification(picpart);
    if(!picpart->nelems()) {
      fprintf(stderr,"%s: empty part on rank %d\n", __func__, rank);
    }
    assert(picpart->nelems());

    delete [] num_ents;

    //****************Convert Safe Tags to the picpart***********
    Omega_h::Write<Omega_h::LO> new_safe(picpart->nelems(), 0, "safe_tag");
    Omega_h::LOs elm_ids = ent_ids[dim];
    const auto convertSafeToPicpart = OMEGA_H_LAMBDA(Omega_h::LO elem_id) {
      const Omega_h::LO new_elem = elm_ids[elem_id];
      if (new_elem >= 0) {
        new_safe[new_elem] = is_safe[elem_id];
      }
    };
    Omega_h::parallel_for(mesh.nelems(), convertSafeToPicpart, "convertSafeToPicpart");
    picpart->add_tag(dim, "safe", 1, Omega_h::LOs(new_safe));
    is_ent_safe = Omega_h::LOs(new_safe);

    commptr = comm;
    /****************Convert gids of each dim to the picpart***********/
    for (int i = 0; i <= dim; ++i) {
      Omega_h::Write<Omega_h::GO> new_ent_gid(picpart->nents(i), 0);
      Omega_h::Write<Omega_h::LO> new_ent_owners(picpart->nents(i), 0);
      Omega_h::Write<Omega_h::LO> new_ent_rlids(picpart->nents(i), 0);
      Omega_h::GOs ent_gid = ent_gid_per_dim[i];
      Omega_h::LOs ent_ids_cpy = ent_ids[i];
      Omega_h::LOs ent_owners = owner_dim[i];
      Omega_h::LOs ent_rlids = rank_lid_per_dim[i];
      const auto convertArraysToPicpart = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
        const Omega_h::LO new_ent = ent_ids_cpy[ent_id];
        //TODO remove this conditional using padding?
        if (new_ent >= 0) {
          new_ent_gid[new_ent] = ent_gid[ent_id];
          new_ent_owners[new_ent] = ent_owners[ent_id];
          new_ent_rlids[new_ent] = ent_rlids[ent_id];
        }
      };
      Omega_h::parallel_for(mesh.nents(i), convertArraysToPicpart, "convertArraysToPicpart");
      global_ids_per_dim[i] = Omega_h::GOs(new_ent_gid);
      rank_lids_per_dim[i] = Omega_h::LOs(new_ent_rlids);
      ent_owner_per_dim[i] = Omega_h::LOs(new_ent_owners);

      //**************** Build communication information ********************//
      Omega_h::LOs picpart_offset_nents = calculateOwnerOffset(new_ent_owners, comm_size);
      setupComm(i, rank_offset_nents[i], picpart_offset_nents, new_ent_owners);
    }
    picpart->add_tag(dim, "ownership", 1, ent_owner_per_dim[dim]);
  }
}

namespace {
  void setOwnerByClassification(Omega_h::Mesh& m, Omega_h::LOs class_owners, int self,
                                Omega_h::Write<Omega_h::LO> owns) {
    auto class_ids = m.get_array<Omega_h::ClassId>(m.dim(), "class_id");
    Omega_h::Write<Omega_h::LO> selfcount(1,0,"selfcount");
    int max_cids = class_owners.size();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto ownByClassification = OMEGA_H_LAMBDA(const Omega_h::LO& id) {
      const Omega_h::ClassId c_id = class_ids[id];
      if (c_id < 0)
        printf("%d Class id is too low %d on entitiy %d\n", rank, c_id, id);
      else if (c_id >= max_cids)
        printf("%d Class id is too high %d on entitiy %d\n", rank, c_id, id);
      owns[id] = class_owners[c_id];
      const auto hasElm = (class_owners[c_id] == self);
      Kokkos::atomic_fetch_add(&(selfcount[0]),hasElm);
    };
    Omega_h::parallel_for(m.nelems(), ownByClassification, "ownByClassification");
    Omega_h::HostWrite<Omega_h::LO> selfcount_h(selfcount);
    if(!selfcount_h[0]) {
      fprintf(stderr, "%s rank %d with no owned elements detected\n", __func__, self);
    }
    assert(selfcount_h[0]);
  }

  //Ownership of lower entity dimensions is determined by the minimum owner of surrounding elements
  Omega_h::LOs defineOwners(Omega_h::Mesh& m, int dim, Omega_h::CommPtr comm,
                            Omega_h::LOs elm_owner) {
    int comm_size = comm->size(), comm_rank = comm->rank();
    Omega_h::Adj ent_to_elm = m.ask_up(dim, m.dim());
    Omega_h::Write<Omega_h::LO> ent_owner(m.nents(dim), "ent_owner");
    auto determineOwner = OMEGA_H_LAMBDA(const Omega_h::LO& ent_id) {
      const auto deg = ent_to_elm.a2ab[ent_id + 1] - ent_to_elm.a2ab[ent_id];
      const auto firstElm = ent_to_elm.a2ab[ent_id];
      Omega_h::LO min = comm_size;
      for (int j = 0; j < deg; ++j) {
        const Omega_h::LO elm = ent_to_elm.ab2b[firstElm+j];
        const Omega_h::LO own = elm_owner[elm];
        if (own < min)
          min = own;
      }
      ent_owner[ent_id] = min;
    };
    Omega_h::parallel_for(m.nents(dim), determineOwner);
    return ent_owner;
  }

  Omega_h::LOs calculateOwnerOffset(Omega_h::LOs owner, int comm_size) {
    Omega_h::Write<Omega_h::LO> offset_nents(comm_size, 0);
    auto countEntsPerRank = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner_index = owner[ent_id];
      Kokkos::atomic_fetch_add(&(offset_nents[owner_index]), 1);
    };
    Omega_h::parallel_for(owner.size(), countEntsPerRank);
    return Omega_h::offset_scan(Omega_h::LOs(offset_nents));
  }

  struct GlobalNumberer {
    using value_type=Omega_h::LO[];
    using size_type=unsigned long;
    size_type value_count;

    Omega_h::LOs rank_offsets;
    Omega_h::Write<Omega_h::GO> elem_gids;
    Omega_h::LOs elem_owner;
    GlobalNumberer(int comm_size, Omega_h::LOs owner,
                   Omega_h::LOs offs, Omega_h::Write<Omega_h::GO> gids) :
      value_count(comm_size), elem_owner(owner), rank_offsets(offs), elem_gids(gids) {}
    OMEGA_H_DEVICE void operator()(const size_type& i, value_type vals,
                                           const bool& final) const {
      const Omega_h::LO own = elem_owner[i];
      if (final) {
        elem_gids[i] = rank_offsets[own] + vals[own];
      }
      ++(vals[own]);
    }
    OMEGA_H_DEVICE void join(volatile value_type update,
                             const volatile value_type input) const {
      for(int i = 0; i < value_count; ++i) {
        update[i] += input[i];
      }
    }
    OMEGA_H_DEVICE void init(value_type vals) const {
      for (size_type i = 0; i < value_count; ++i) {
        vals[i] = 0;
      }
    }
  };
  Omega_h::LOs createGlobalNumbering(Omega_h::LOs owner, int comm_size,
                                     Omega_h::Write<Omega_h::GO> elem_gid) {
    Omega_h::LOs rank_offset_nelms = calculateOwnerOffset(owner, comm_size);

    //Globally number the elements
    GlobalNumberer gnr(comm_size, owner, rank_offset_nelms, elem_gid);
    Omega_h::parallel_scan(owner.size(), gnr);
    return rank_offset_nelms;
  }

  Omega_h::LOs rankLidNumbering(Omega_h::LOs owners, Omega_h::LOs offset, Omega_h::GOs gids) {
    Omega_h::LO nents = gids.size();
    Omega_h::Write<Omega_h::LO> ent_rank_lids(nents,0);
    auto calculateRankLids = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner = owners[ent_id];
      ent_rank_lids[ent_id] = gids[ent_id] - offset[owner];
    };
    Omega_h::parallel_for(nents, calculateRankLids, "calculateRankLids");
    return ent_rank_lids;
  }

  void bfsBufferLayers(Omega_h::Mesh& mesh, int bridge_dim, Omega_h::CommPtr comm,
                       int safe_layers, int ghost_layers,
                       Omega_h::Write<Omega_h::LO> is_safe,
                       Omega_h::LOs owner, Omega_h::Write<Omega_h::LO> has_part) {
    int rank = comm->rank();
    int comm_size = comm->size();
    Omega_h::Write<Omega_h::LO> is_visited(mesh.nelems());
    Omega_h::Write<Omega_h::LO> is_visited_next(mesh.nelems());
    const auto initVisit = OMEGA_H_LAMBDA( Omega_h::LO elem_id){
      is_visited[elem_id] = is_safe[elem_id] = (owner[elem_id] == rank);
      is_visited_next[elem_id] = is_visited[elem_id];
    };
    Omega_h::parallel_for(mesh.nelems(), initVisit, "initVisit");
    auto initHasPart = OMEGA_H_LAMBDA(Omega_h::LO i) {
      has_part[i] = (i == rank);
    };
    Omega_h::parallel_for(comm_size, initHasPart);

    const auto bridge2elems = mesh.ask_up(bridge_dim, mesh.dim());
    auto ghostingBFS = OMEGA_H_LAMBDA( Omega_h::LO bridge_id) {
      const auto deg = bridge2elems.a2ab[bridge_id + 1] - bridge2elems.a2ab[bridge_id];
      const auto firstElm = bridge2elems.a2ab[bridge_id];
      bool is_visited_here = false;
      for (int j = 0; j < deg; ++j) {
        const auto elm = bridge2elems.ab2b[firstElm+j];
        if (is_visited[elm])
          is_visited_here = 1;
      }
      for (int j = 0; j < deg*is_visited_here; ++j) {
        const auto elm = bridge2elems.ab2b[firstElm+j];
        is_visited_next[elm] = true;
      }
    };
    for (int i = 0; i < ghost_layers || i < safe_layers; ++i) {
      Omega_h::parallel_for(mesh.nents(bridge_dim), ghostingBFS,"ghostingBFS");
      auto copyVisit = OMEGA_H_LAMBDA( Omega_h::LO elm_id) {
        is_visited[elm_id] = is_visited_next[elm_id];
        if (i < safe_layers)
          is_safe[elm_id] = is_visited_next[elm_id];
        if (i < ghost_layers)
          if (is_visited[elm_id])
            has_part[owner[elm_id]] = 1;
      };

      Omega_h::parallel_for(mesh.nelems(), copyVisit, "copyVisit");
    }
  }

  void setSafeEnts(Omega_h::Mesh& mesh, int dim, int size, Omega_h::Write<Omega_h::LO> has_part,
                   Omega_h::LOs owner, Omega_h::Write<Omega_h::LO> buf) {
    if (dim == mesh.dim()) {
      auto setSafeEntsL = OMEGA_H_LAMBDA( Omega_h::LO elm_id) {
        buf[elm_id] = has_part[owner[elm_id]];
      };
      Omega_h::parallel_for(size, setSafeEntsL, "setSafeEnts");
    }
    else {
      const Omega_h::Adj downAdj = mesh.ask_down(mesh.dim(),dim);
      auto deg = Omega_h::element_degree(mesh.family(), mesh.dim(), dim);
      auto setSafeEnts = OMEGA_H_LAMBDA( Omega_h::LO elm_id) {
        bool is_buffered = has_part[owner[elm_id]];
        const auto firstEnt = elm_id * deg;
        for (int j = 0; j < deg; ++j) {
          const auto ent = downAdj.ab2b[firstEnt+j];
          if( is_buffered )
            buf[ent] = 1;
        }
      };
      Omega_h::parallel_for(size, setSafeEnts, "setSafeEnts");
    }
  }

  //TODO Replace with omega_h reduce/scan
  Omega_h::LO sumPositives(Omega_h::LO size, Omega_h::Write<Omega_h::LO> arr) {
    Omega_h::LO sum = 0;
    Kokkos::parallel_reduce(size, OMEGA_H_LAMBDA(const int i, Omega_h::LO& lsum) {
        lsum += arr[i] > 0 ;
      }, sum);
    return sum;
  }
  void gatherCoords(Omega_h::Mesh& mesh, Omega_h::LOs vert_ids,
                    Omega_h::Write<Omega_h::Real> new_coords) {
    const auto meshDim = mesh.dim();
    Omega_h::Reals n2c = mesh.coords();
    auto gatherCoordsL = OMEGA_H_LAMBDA(Omega_h::LO vert_id)  {
      const Omega_h::LO first_vert = vert_id * meshDim;
      const Omega_h::LO first_new_vert = vert_ids[vert_id]*meshDim;
      const int deg = (first_new_vert >=0 ) * meshDim;
      for (int i = 0; i < deg; ++i)
        new_coords[first_new_vert + i] = n2c[first_vert + i];
    };
    Omega_h::parallel_for(mesh.nverts(), gatherCoordsL, "gatherCoords");
  }

  void buildAndClassify(Omega_h::Mesh& full_mesh, Omega_h::Mesh* picpart, int dim, int num_ents,
                        Omega_h::LOs ent_ids, Omega_h::LOs vert_ids,
                        Omega_h::Write<Omega_h::Real> new_coords) {
    Omega_h::Write<Omega_h::LO> ent2v((num_ents) * (dim + 1));
    Omega_h::Write<Omega_h::LO> ent_class(num_ents);
    auto old_class = full_mesh.get_array<Omega_h::ClassId>(dim, "class_id");
    if (dim != 0) {
      //All entities of dimension > 0
      const auto ent2vert = full_mesh.ask_down(dim, 0);

      auto getDownAndClass = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
        const Omega_h::LO new_ent = ent_ids[ent_id];
        const int nvpe = dim + 1;
        const Omega_h::LO first_vert = ent_id * nvpe;
        const Omega_h::LO first_new_vert = new_ent * nvpe;
        if (new_ent >= 0) {
          for (int j = 0; j < nvpe; ++j) {
            const Omega_h::LO old_vert = ent2vert.ab2b[first_vert+j];
            const Omega_h::LO new_v = first_new_vert + j;
            const Omega_h::LO new_v_id = vert_ids[old_vert];
            ent2v[new_v] = new_v_id;
          }
          ent_class[new_ent] = old_class[ent_id];
        }
      };
      Omega_h::parallel_for(full_mesh.nents(dim), getDownAndClass, "getDownAndClass");
    }
    else {
      //We cant do downwards with vertices
      auto getVertClass = OMEGA_H_LAMBDA(Omega_h::LO vert_id) {
        const Omega_h::LO new_vert = ent_ids[vert_id];
        if (new_vert >= 0) {
          ent_class[new_vert] = old_class[vert_id];
          ent2v[new_vert] = new_vert;
        }
      };
      Omega_h::parallel_for(full_mesh.nverts(), getVertClass, "getVertClass");
    }
    if (dim == full_mesh.dim())
      Omega_h::build_from_elems_and_coords(picpart, full_mesh.family(), full_mesh.dim(),
                                         ent2v, new_coords);
    Omega_h::classify_equal_order(picpart, dim, ent2v, ent_class);
  }
}
