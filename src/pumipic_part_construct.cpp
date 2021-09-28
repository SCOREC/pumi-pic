#include "pumipic_mesh.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_adj.hpp> //reflect_down
#include <Omega_h_build.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_scan.hpp>
#include <Omega_h_file.hpp>
#include "pumipic_lb.hpp"

namespace {
  template <class T>
  void writeArray(Omega_h::Read<T> arr, std::string name) {
    Omega_h::HostRead<T> arr_hr(arr);
    Omega_h::LO const nl = arr_hr.size();
    std::cout << name << " size " << nl << "\n";
    for(int l=0; l<nl; l++) {
      const auto d = arr_hr[l];
      std::cout << l << " " << (int)d << "\n";
    }
  }

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
  void bfsSafeInward(Omega_h::Mesh& mesh, int bridge_dim, Omega_h::CommPtr comm,
                     int safe_layers, Omega_h::LOs owner, Omega_h::LOs has_part,
                     Omega_h::Write<Omega_h::LO> safe);
  void setSafeEnts(Omega_h::Mesh& mesh, int dim, int size, Omega_h::Write<Omega_h::LO> has_part,
                   Omega_h::LOs owner, Omega_h::Write<Omega_h::LO> buf);
  Omega_h::LO sumPositives(Omega_h::LO size, Omega_h::Write<Omega_h::LO> arr);
  void gatherCoords(Omega_h::Mesh& mesh, Omega_h::LOs vert_ids,
                    Omega_h::Write<Omega_h::Real> new_coords);
  void buildAndClassify(Omega_h::Mesh& full_mesh, Omega_h::Mesh* picpart, int dim, int num_ents,
                        Omega_h::LOs ent_ids, Omega_h::LOs vert_ids,
                        Omega_h::Write<Omega_h::Real> new_coords);
  void buildAndClassifyFull2pp(Omega_h::Mesh& full_mesh, Omega_h::Mesh& picpart,
      int dim, int pp_num_ents, Omega_h::LOs full2pp_entIds, Omega_h::LOs full2pp_downIds);
  void classifyVerts(Omega_h::Mesh& full_mesh, Omega_h::Mesh& picpart,
      const int pp_num_verts, Omega_h::LOs full2pp_entIds);
  template <class T>
  void convertTag(Omega_h::Mesh full_mesh, Omega_h::Mesh* picpart, int dim,
                  Omega_h::LOs entToEnt, Omega_h::TagBase const* tag,
                  const char* new_name = "");
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

  Mesh::Mesh(Omega_h::Mesh& mesh, Omega_h::LOs owner, int ghost_layers, int safe_layers) {
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

    if (in.bufferMethod == Input::BFS && in.safeMethod == Input::FULL) {
      bfsSafeInward(in.m,in.bridge_dim, comm, in.safeBFSLayers, owners, Omega_h::LOs(has_part),
                    is_safe);
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

    //Get global entity counts from full mesh
    num_entites[3] = 0;
    for (int i = 0; i <= dim; ++i) {
      num_entites[i] = mesh.nents(i);
    }

    /************* Define Ownership of each lower dimension entity ************/
    Omega_h::LOs owner_dim[4];
    for (int i = 0; i < dim; ++i) {
      owner_dim[i] = defineOwners(mesh, i, comm, owner);
      mesh.add_tag(i, "ownership", 1, owner_dim[i]);
    }
    owner_dim[dim] = owner;
    mesh.add_tag(dim, "ownership", 1, owner_dim[dim]);

    mesh.add_tag(dim, "safe", 1, Omega_h::LOs(is_safe));
    if (render && rank == 0)
      Omega_h::vtk::write_parallel("partition", &mesh, dim);
    /************* Globally Number Entities **********/
    Omega_h::LOs rank_lid_per_dim[4];
    Omega_h::LOs rank_offset_nents[4];
    for (int i = 0; i <= dim; ++i) {
      Omega_h::Write<Omega_h::GO> gids(mesh.nents(i), "global_ids");
      rank_offset_nents[i] = createGlobalNumbering(owner_dim[i], comm_size, gids);
      auto ent_gids = Omega_h::GOs(gids);
      mesh.add_tag(i, "gids", 1, ent_gids);
      auto rank_lids = rankLidNumbering(owner_dim[i], rank_offset_nents[i],
                                        ent_gids);
      mesh.add_tag(i, "rank_lids", 1, rank_lids);
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
    for (int i = 0; i <= dim; ++i) {
      setSafeEnts(mesh, i, mesh.nelems(), has_part, owner_dim[dim], buf_ents[i]);
      std::stringstream ss;
      ss << "buf_ents[" << i <<"]";
      std::string str = ss.str();
      writeArray(Omega_h::Read<Omega_h::LO>(buf_ents[i]), str); //all entries should be '1'
    }

    //Gather number of entities remaining in the picpart
    Omega_h::GO* num_ents = new Omega_h::GO[dim+1];
    for (int i = 0; i <= dim; ++i) {
      num_ents[i] = sumPositives(mesh.nents(i), buf_ents[i]);
      std::cerr << "num_ents[" << i << "] " << num_ents[i] << " mesh.nents(i) " << mesh.nents(i) << "\n";
    }

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
      //std::stringstream ss;
      //ss << "ent_ids[" << i <<"]";
      //std::string str = ss.str();
      //writeArray(Omega_h::Read<Omega_h::LO>(ent_ids[i]), str);
    }

    //If full mesh buffer then we don't need to make new mesh for the picparts
    if (isFullMesh()) {
      //Set picpart to point to the mesh
      picpart = &mesh;

      //Copy global numbering to global_serial for consistency
      for (int i = 0; i <= dim; ++i) {
        if (mesh.has_tag(i, "global")) {
          Omega_h::GOs tag_array = mesh.get_array<Omega_h::GO>(i, "global");
          mesh.add_tag(i, "global_serial", 1, tag_array);
        }
      }
    }
    //************Build a new mesh as the picpart**************
    else {
      const auto lib = mesh.library();
      picpart = new Omega_h::Mesh(lib);
      picpart->set_comm(lib->self());
      picpart->set_dim(dim);
      picpart->set_family(OMEGA_H_SIMPLEX);
      picpart->set_parting(OMEGA_H_ELEM_BASED);

      //Gather coordinates
      Omega_h::Write<Omega_h::Real> new_coords((num_ents[0])*dim,0);
      gatherCoords(mesh, ent_ids[0], new_coords);

      //Build the mesh
      picpart->set_verts(num_ents[0]);
      for (int i = 1; i <= dim; ++i)
        buildAndClassifyFull2pp(mesh, *picpart, i, num_ents[i], ent_ids[i], ent_ids[0]);
      classifyVerts(mesh, *picpart, num_ents[0], ent_ids[0]);
      picpart->add_coords(new_coords);
      Omega_h::finalize_classification(picpart);
      if(!picpart->nelems()) {
        fprintf(stderr,"%s: empty part on rank %d\n", __func__, rank);
      }
      assert(picpart->nelems());

      /****************Convert all tags to picparts****************/
      for (int i = 0; i <= dim; ++i) {
        //Move tags from old mesh to new mesh
        for (int j = 0; j < mesh.ntags(i); ++j) {
          Omega_h::TagBase const* tagbase = mesh.get_tag(i,j);
          // Ignore Omega_h internal tags
          if (tagbase->name() == "coordinates" ||
              tagbase->name() == "class_id" ||
              tagbase->name() == "class_dim")
            continue;
          if (tagbase->name() == "global")
            convertTag<Omega_h::I64>(mesh, picpart, i, ent_ids[i], tagbase,
                                     "global_serial");
          if (tagbase->type() == OMEGA_H_I8)
            convertTag<Omega_h::I8>(mesh, picpart, i, ent_ids[i], tagbase);
          if (tagbase->type() == OMEGA_H_I32)
            convertTag<Omega_h::I32>(mesh, picpart, i, ent_ids[i], tagbase);
          if (tagbase->type() == OMEGA_H_I64)
            convertTag<Omega_h::I64>(mesh, picpart, i, ent_ids[i], tagbase);
          if (tagbase->type() == OMEGA_H_F64)
            convertTag<Omega_h::Real>(mesh, picpart, i, ent_ids[i], tagbase);
        }
      }
    }

    delete [] num_ents;
    commptr = comm;

    //**************** Build communication information ********************//
    for (int i = 0; i <= dim; ++i) {
      Omega_h::LOs picpart_offset_nents = calculateOwnerOffset(entOwners(i), comm_size);
      setupComm(i, rank_offset_nents[i], picpart_offset_nents, entOwners(i));
    }

    //Create load balancer
    ptcl_balancer = new ParticleBalancer(*this);

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

  void BFS(int nents, Omega_h::Adj bridge2elems,Omega_h::LOs visited,
           Omega_h::Write<Omega_h::LO> visited_next) {
    auto meshBFS = OMEGA_H_LAMBDA(const Omega_h::LO bridge_id) {
      const auto deg = bridge2elems.a2ab[bridge_id + 1] - bridge2elems.a2ab[bridge_id];
      const auto firstElm = bridge2elems.a2ab[bridge_id];
      bool is_visited_here = false;
      for (int j = 0; j < deg; ++j) {
        const auto elm = bridge2elems.ab2b[firstElm+j];
        if (visited[elm])
          is_visited_here = true;
      }
      const int loops = deg*is_visited_here;
      for (int j = 0; j < loops; ++j) {
        const auto elm = bridge2elems.ab2b[firstElm+j];
        visited_next[elm] = true;
      }
    };
    Omega_h::parallel_for(nents, meshBFS,"meshBFS");
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
    auto initSelfPart = OMEGA_H_LAMBDA(Omega_h::LO i) {
      has_part[rank] = true;
    };
    Omega_h::parallel_for(1, initSelfPart);

    const auto bridge2elems = mesh.ask_up(bridge_dim, mesh.dim());
    for (int i = 0; i < ghost_layers || i < safe_layers; ++i) {
      BFS(mesh.nents(bridge_dim), bridge2elems, Omega_h::LOs(is_visited), is_visited_next);
      auto copyVisit = OMEGA_H_LAMBDA( Omega_h::LO elm_id) {
        is_visited[elm_id] = is_visited_next[elm_id];
        if (i == safe_layers - 1)
          is_safe[elm_id] = is_visited_next[elm_id];
        if (i < ghost_layers)
          if (is_visited[elm_id])
            has_part[owner[elm_id]] = 1;
      };
      Omega_h::parallel_for(mesh.nelems(), copyVisit, "copyVisit");
    }
  }

  void bfsSafeInward(Omega_h::Mesh& mesh, int bridge_dim, Omega_h::CommPtr comm,
                     int safe_layers, Omega_h::LOs owner, Omega_h::LOs has_part,
                     Omega_h::Write<Omega_h::LO> safe) {
    Omega_h::Write<Omega_h::LO> is_visited(mesh.nelems(), 0);
    Omega_h::Write<Omega_h::LO> is_visited_next(mesh.nelems(), 0);
    const auto initVisit = OMEGA_H_LAMBDA( Omega_h::LO elem_id) {
      const Omega_h::LO own = owner[elem_id];
      is_visited[elem_id] = is_visited_next[elem_id] = !has_part[own];
    };
    Omega_h::parallel_for(mesh.nelems(), initVisit, "initVisit");

    const auto bridge2elems = mesh.ask_up(bridge_dim, mesh.dim());
    for (int i = 0; i < safe_layers; ++i) {
      BFS(mesh.nents(bridge_dim), bridge2elems, Omega_h::LOs(is_visited), is_visited_next);
      auto copyVisit = OMEGA_H_LAMBDA(Omega_h::LO elm_id) {
        is_visited[elm_id] = is_visited_next[elm_id];
      };
      Omega_h::parallel_for(mesh.nelems(), copyVisit, "copyVisit");
    }
    int rank = comm->rank();
    auto setSafe = OMEGA_H_LAMBDA(Omega_h::LO elm_id) {
      const Omega_h::LO visit = is_visited[elm_id];
      const Omega_h::LO own = (owner[elm_id] == rank);
      safe[elm_id] = !visit || own;
    };
    Omega_h::parallel_for(mesh.nelems(), setSafe, "setSafe");
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

  Omega_h::Adj getAdj(const int dim, Omega_h::LOs down, Omega_h::Mesh m) {
    if(dim == 1) return Omega_h::Adj(down);
    else if(dim == 2) {
      const auto e2v = m.get_adj(1,0);
      const auto v2e = m.ask_up(0,1);
      const auto adj = Omega_h::reflect_down(down, e2v.ab2b, v2e, m.family(), dim, dim-1); //fails here
      return adj;
    }
    else exit(EXIT_FAILURE);
  }

  void buildAndClassifyFull2pp(Omega_h::Mesh& full_mesh, Omega_h::Mesh& picpart,
      int dim, int pp_num_ents, Omega_h::LOs full2pp_entIds, Omega_h::LOs full2pp_downIds) {
    //extract downward adjaceny arrays from full_mesh and pass them into the new
    //picpart mesh via Omega_h::set_ents(...)
    auto full_classId = full_mesh.get_array<Omega_h::ClassId>(dim, "class_id");
    auto full_classDim = full_mesh.get_array<Omega_h::I8>(dim, "class_dim");
    const auto degree = Omega_h::element_degree(full_mesh.family(), dim, 0);
    Omega_h::Write<Omega_h::LO> ppDown(pp_num_ents*degree);
    Omega_h::Write<Omega_h::ClassId> ppClassId(pp_num_ents);
    Omega_h::Write<Omega_h::I8> ppClassDim(pp_num_ents);
    const auto downFull = full_mesh.ask_down(dim,0);

    auto getDownAndClass = OMEGA_H_LAMBDA(Omega_h::LO full_ent_id) {
      const Omega_h::LO pp_ent = full2pp_entIds[full_ent_id];
      const Omega_h::LO full_first_down = full_ent_id * degree;
      const Omega_h::LO pp_first_down = pp_ent * degree;
      if (pp_ent >= 0) {
        //fill in the downward adjacency array
        for (int j = 0; j < degree; ++j) {
          const Omega_h::LO full_down = downFull.ab2b[full_first_down+j];
          const Omega_h::LO pp_down = pp_first_down + j;
          const Omega_h::LO pp_down_id = full2pp_downIds[full_down];
          ppDown[pp_down] = pp_down_id;
        }
        //transfer classification
        ppClassId[pp_ent] = full_classId[full_ent_id];
        ppClassDim[pp_ent] = full_classDim[full_ent_id];
      } else {
        printf("entity %d %d is not new\n", dim, full_ent_id);
      }
    };
    Omega_h::parallel_for(full_mesh.nents(dim), getDownAndClass, "getDownAndClass");

    const auto adj = getAdj(dim, ppDown, picpart);
    picpart.set_ents(dim,adj);
    
    //set classification
    picpart.add_tag<Omega_h::ClassId>(dim, "class_id", 1, Omega_h::Read<Omega_h::ClassId>(ppClassId));
    picpart.add_tag<Omega_h::I8>(dim, "class_dim", 1, Omega_h::Read<Omega_h::I8>(ppClassDim));
  }

  void classifyVerts(Omega_h::Mesh& full_mesh, Omega_h::Mesh& picpart,
      const int pp_num_verts, Omega_h::LOs full2pp_entIds) {
    const auto vdim = 0;
    auto full_classId = full_mesh.get_array<Omega_h::ClassId>(vdim, "class_id");
    auto full_classDim = full_mesh.get_array<Omega_h::I8>(vdim, "class_dim");
    Omega_h::Write<Omega_h::ClassId> ppClassId(pp_num_verts);
    Omega_h::Write<Omega_h::I8> ppClassDim(pp_num_verts);

    auto getClass = OMEGA_H_LAMBDA(Omega_h::LO full_ent_id) {
      const Omega_h::LO pp_ent = full2pp_entIds[full_ent_id];
      if (pp_ent >= 0) {
        ppClassId[pp_ent] = full_classId[full_ent_id];
        ppClassDim[pp_ent] = full_classDim[full_ent_id];
      } else {
        printf("entity %d %d is not new\n", vdim, full_ent_id);
      }
    };
    Omega_h::parallel_for(pp_num_verts, getClass, "getClass");
    //set classification
    picpart.add_tag<Omega_h::ClassId>(vdim, "class_id", 1, Omega_h::Read<Omega_h::ClassId>(ppClassId));
    picpart.add_tag<Omega_h::I8>(vdim, "class_dim", 1, Omega_h::Read<Omega_h::I8>(ppClassDim));
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
        } else {
          printf("entity %d %d is not new\n", dim, ent_id);
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

  template <class T>
  void convertTag(Omega_h::Mesh full_mesh, Omega_h::Mesh* picpart, int dim,
                  Omega_h::LOs entToEnt, Omega_h::TagBase const* tagbase,
                  const char* new_name) {
    Omega_h::Read<T> tag = full_mesh.get_array<T>(dim, tagbase->name());
    Omega_h::LO nents = full_mesh.nents(dim);
    int nvalues = tag.size() / nents;
    Omega_h::Write<T> new_tag(picpart->nents(dim) * nvalues);
    auto convertTagValues = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO new_ent = entToEnt[ent_id];
      if (new_ent >= 0) {
        for (int i = 0; i < nvalues; ++i)
          new_tag[new_ent*nvalues + i] = tag[ent_id*nvalues + i];
      }
    };
    Omega_h::parallel_for(nents, convertTagValues, "convertTagValues");
    if (strlen(new_name) == 0)
      picpart->add_tag(dim, tagbase->name(), nvalues, Omega_h::Read<T>(new_tag));
    else
      picpart->add_tag(dim, new_name, nvalues, Omega_h::Read<T>(new_tag));
  }
}
