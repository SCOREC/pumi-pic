#include "pumipic_mesh.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_int_scan.hpp>

namespace {
  Omega_h::LOs calculateOwnerOffset(Omega_h::Write<Omega_h::LO> owner, int comm_size);
  Omega_h::LOs createGlobalNumbering(Omega_h::Write<Omega_h::LO> owner, int comm_size,
                             Omega_h::Write<Omega_h::GO> elem_gid);
  void bfsBufferLayers(Omega_h::Mesh& mesh, int bridge_dim, int safe_layers, int ghost_layers,
                       Omega_h::Write<Omega_h::LO> is_safe, Omega_h::Write<Omega_h::LO> is_visited,
                       Omega_h::Write<Omega_h::LO> owner, Omega_h::Write<Omega_h::LO> has_part);
  void setSafeEnts(Omega_h::Mesh& mesh, int dim, int size, Omega_h::Write<Omega_h::LO> has_part, 
                   Omega_h::Write<Omega_h::LO> owner, Omega_h::Write<Omega_h::LO> buf);
  Omega_h::LO sumPositives(Omega_h::LO size, Omega_h::Write<Omega_h::LO> arr);
  // void numberValidEntries(Omega_h::LO size, Omega_h::Write<Omega_h::LO> is_valid,
  //                         Omega_h::Write<Omega_h::LO> numbering);
  void gatherCoords(Omega_h::Mesh& mesh, Omega_h::LOs vert_ids,
                    Omega_h::Write<Omega_h::Real> new_coords);
  void buildAndClassify(Omega_h::Mesh& full_mesh, Omega_h::Mesh* picpart, int dim, int num_ents,
                        Omega_h::LOs ent_ids, Omega_h::LOs vert_ids,
                        Omega_h::Write<Omega_h::Real> new_coords);
}

namespace pumipic {
    Mesh::Mesh(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    /************* Globally Number Element **********/
    Omega_h::Write<Omega_h::GO> elem_gid(mesh.nelems());
    Omega_h::LOs rank_offset_nelms = createGlobalNumbering(owner, comm_size, elem_gid);
    //TODO define global numbering on all entity types

    // *********** Set safe zone and buffer to be entire mesh**************** //
    Omega_h::Write<Omega_h::LO> is_safe(mesh.nelems(), 1);
    Omega_h::Write<Omega_h::LO> has_part(comm_size, 1);

    constructPICPart(mesh, owner, elem_gid, rank_offset_nelms, has_part, is_safe);
  }

  Mesh::Mesh(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner,
                  int ghost_layers, int safe_layers) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (ghost_layers < safe_layers) {
      if (!rank)
        fprintf(stderr, "Ghost layers must be >= safe layers");
      throw 1;
    }

    /************* Globally Number Element **********/
    Omega_h::Write<Omega_h::GO> elem_gid(mesh.nelems());
    Omega_h::LOs rank_offset_nelms = createGlobalNumbering(owner, comm_size, elem_gid);
    //TODO define global numbering on all entity types

    // **********Determine safe zone and ghost region**************** //
    Omega_h::Write<Omega_h::LO> is_safe(mesh.nelems());
    Omega_h::Write<Omega_h::LO> is_visited(mesh.nelems());
    Omega_h::Write<Omega_h::LO> has_part(comm_size);
    int bridge_dim = 0;
    bfsBufferLayers(mesh, bridge_dim, safe_layers, ghost_layers, is_safe, is_visited, 
                    owner, has_part);

    constructPICPart(mesh, owner, elem_gid, rank_offset_nelms, has_part, is_safe);
  }
  void Mesh::constructPICPart(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner,
                              Omega_h::Write<Omega_h::GO> elem_gid,
                              Omega_h::LOs rank_offset_nelms,
                              Omega_h::Write<Omega_h::LO> has_part,
                              Omega_h::Write<Omega_h::LO> is_safe) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int dim = mesh.dim();

    /***************** Count the number of parts in the picpart ****************/
    num_cores[dim] = sumPositives(has_part.size(),has_part) - 1;

    
    /***************** Count Number of Entities in the PICpart *************/
    //Mark all entities owned by a part with has_part[part] = true as staying
    Omega_h::Write<Omega_h::LO> buf_ents[4];
    for (int i = 0; i <= dim; ++i) 
      buf_ents[i] = Omega_h::Write<Omega_h::LO>(mesh.nents(i),0);
    for (int i = 0; i <= dim; ++i)
      setSafeEnts(mesh, i, mesh.nelems(), has_part, owner, buf_ents[i]);

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

    delete [] num_ents;

    //****************Convert Tags to the picpart***********
    Omega_h::Write<Omega_h::LO> new_safe(picpart->nelems(), 0);
    Omega_h::Write<Omega_h::GO> new_elem_gid(picpart->nelems(), 0);
    Omega_h::Write<Omega_h::LO> new_ent_owners(picpart->nelems(), 0);
    Omega_h::LOs elm_ids = ent_ids[dim];
    const auto convertArraysToPicpart = OMEGA_H_LAMBDA(Omega_h::LO elem_id) {
      const Omega_h::LO new_elem = elm_ids[elem_id];
      //TODO remove this conditional using padding?
      if (new_elem >= 0) {
        new_safe[new_elem] = is_safe[elem_id];
        new_elem_gid[new_elem] = elem_gid[elem_id];
        new_ent_owners[new_elem] = owner[elem_id];
      }
    };
    Omega_h::parallel_for(mesh.nelems(), convertArraysToPicpart, "convertArraysToPicpart");
    picpart->add_tag(dim, "safe", 1, Omega_h::Read<Omega_h::LO>(new_safe));
    global_ids_per_dim[dim] = Omega_h::Read<Omega_h::GO>(new_elem_gid);
    is_ent_safe = Omega_h::LOs(new_safe);

    //**************** Build communication information ********************//
    //TODO create communication information for each entity dimension
    commptr = lib->world();
    Omega_h::LOs picpart_offset_nelms = calculateOwnerOffset(new_ent_owners, comm_size);
    setupComm(3, rank_offset_nelms, picpart_offset_nelms, new_ent_owners);
  }
}

namespace {

  Omega_h::LOs calculateOwnerOffset(Omega_h::Write<Omega_h::LO> owner, int comm_size) {
    Omega_h::Write<Omega_h::LO> offset_nents(comm_size, 0);
    auto countEntsPerRank = OMEGA_H_LAMBDA(Omega_h::LO ent_id) {
      const Omega_h::LO owner_index = owner[ent_id];
      Kokkos::atomic_fetch_add(&(offset_nents[owner_index]), 1);
    };
    Omega_h::parallel_for(owner.size(), countEntsPerRank);
    return Omega_h::offset_scan(Omega_h::LOs(offset_nents));
  }

  Omega_h::LOs createGlobalNumbering(Omega_h::Write<Omega_h::LO> owner, int comm_size,
                             Omega_h::Write<Omega_h::GO> elem_gid) {
    Omega_h::LOs rank_offset_nelms = calculateOwnerOffset(owner, comm_size);

    //Globally number the elements
    Omega_h::Write<Omega_h::LO> elem_gid_rank = Omega_h::deep_copy(rank_offset_nelms);
    auto setElemGID = OMEGA_H_LAMBDA(Omega_h::LO elm_id) {
      elem_gid[elm_id] = Kokkos::atomic_fetch_add(&(elem_gid_rank[owner[elm_id]]), 1);
    };
    Omega_h::parallel_for(owner.size(), setElemGID);
    return rank_offset_nelms;
  }

  void bfsBufferLayers(Omega_h::Mesh& mesh, int bridge_dim, int safe_layers, int ghost_layers,
                       Omega_h::Write<Omega_h::LO> is_safe, Omega_h::Write<Omega_h::LO> is_visited,
                       Omega_h::Write<Omega_h::LO> owner, Omega_h::Write<Omega_h::LO> has_part) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
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
    auto copySafe = OMEGA_H_LAMBDA( Omega_h::LO elm_id) {
      is_safe[elm_id] = is_visited_next[elm_id];
    };
    auto copyVisit = OMEGA_H_LAMBDA( Omega_h::LO elm_id) {
      is_visited[elm_id] = is_visited_next[elm_id];
      if (is_visited[elm_id])
        has_part[owner[elm_id]] = 1;
    };
    for (int i = 0; i < ghost_layers; ++i) {
      Omega_h::parallel_for(mesh.nents(bridge_dim), ghostingBFS,"ghostingBFS");
      if (i < safe_layers)
        Omega_h::parallel_for(mesh.nelems(), copySafe, "copySafe");
      Omega_h::parallel_for(mesh.nelems(), copyVisit, "copyVisit");
    }
  }

  void setSafeEnts(Omega_h::Mesh& mesh, int dim, int size, Omega_h::Write<Omega_h::LO> has_part, 
                   Omega_h::Write<Omega_h::LO> owner, Omega_h::Write<Omega_h::LO> buf) {
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
  // void numberValidEntries(Omega_h::LO size, Omega_h::Write<Omega_h::LO> is_valid, 
  //                         Omega_h::Write<Omega_h::LO> numbering) {
  //   Kokkos::parallel_scan(size, KOKKOS_LAMBDA(const int& i, int& num, const bool& final) {
  //     num += is_valid[i];
  //     numbering[i] += num * final * is_valid[i];
  //   });
  // }
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
