#include "pumipic_mesh.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <Omega_h_tag.hpp>
#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_reduce.hpp>
#include <Omega_h_build.hpp>
#include <Kokkos_Core.hpp>

namespace {
  void calculateOwnerOffset(Omega_h::Write<Omega_h::LO>& owner,
                            Omega_h::Write<Omega_h::LO>& offset_nents);
  void createGlobalNumbering(Omega_h::Write<Omega_h::LO>& owner,
                             Omega_h::Write<Omega_h::LO>& rank_offset_nelms,
                             Omega_h::Write<Omega_h::GO>& elem_gid);
  void bfsBufferLayers(Omega_h::Mesh& mesh, int bridge_dim, int safe_layers, int ghost_layers,
                       Omega_h::Write<Omega_h::LO>& is_safe, Omega_h::Write<Omega_h::LO>& is_visited,
                       Omega_h::Write<Omega_h::LO>& owner, Omega_h::Write<Omega_h::LO>& has_part);
  void setSafeEnts(Omega_h::Mesh& mesh, int dim, int size, Omega_h::Write<Omega_h::LO>& has_part, 
                   Omega_h::Write<Omega_h::LO>& owner, Omega_h::Write<Omega_h::LO>& buf);
  Omega_h::LO sumPositives(Omega_h::LO size, Omega_h::Write<Omega_h::LO>& arr);
  void numberValidEntries(Omega_h::LO size, Omega_h::Write<Omega_h::LO>& is_valid,
                          Omega_h::Write<Omega_h::LO>& numbering);
  void gatherCoords(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO>& vert_ids,
                    Omega_h::Write<Omega_h::Real>& new_coords);
  void buildAndClassify(Omega_h::Mesh& full_mesh, Omega_h::Mesh* picpart, int dim, int num_ents,
                        Omega_h::Write<Omega_h::LO>& ent_ids, Omega_h::Write<Omega_h::LO>& vert_ids,
                        Omega_h::Write<Omega_h::Real>& new_coords);
}

namespace pumipic {
  Mesh::Mesh(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO>& owner,
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
    Omega_h::Write<Omega_h::LO> rank_offset_nelms(comm_size+1,0);
    createGlobalNumbering(owner, rank_offset_nelms, elem_gid);
    //TODO define global numbering on all entity types

    // **********Determine safe zone and ghost region**************** //
    Omega_h::Write<Omega_h::LO> is_safe(mesh.nelems());
    Omega_h::Write<Omega_h::LO> is_visited(mesh.nelems());
    const auto initVisit = OMEGA_H_LAMBDA( Omega_h::LO elem_id){
      is_visited[elem_id] = is_safe[elem_id] = (owner[elem_id] == rank);
    };
    Omega_h::parallel_for(mesh.nelems(), initVisit, "initVisit");
    Omega_h::Write<Omega_h::LO> has_part(comm_size);
    for (int i = 0; i < comm_size; ++i)
      has_part[i] = (i == rank);
    int bridge_dim = 0;
    bfsBufferLayers(mesh, bridge_dim, safe_layers, ghost_layers, is_safe, is_visited, 
                    owner, has_part);

    constructPICPart(mesh, owner, elem_gid, rank_offset_nelms, has_part, is_safe);
  }
  void Mesh::constructPICPart(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO>& owner,
                              Omega_h::Write<Omega_h::GO> elem_gid,
                              Omega_h::Write<Omega_h::LO> rank_offset_nelms,
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
    Omega_h::Write<Omega_h::LO> num_ents(dim+1,0);
    for (int i = 0; i <= dim; ++i)
      num_ents[i] = sumPositives(mesh.nents(i), buf_ents[i]);

    /**************** Create numberings for the entities on the picpart **************/
    Omega_h::Write<Omega_h::LO> ent_ids[4];
    for (int i = 0; i <= dim; ++i) {
      //Default the value to the number of entities in the pic part (for padding)
      ent_ids[i] = Omega_h::Write<Omega_h::LO>(mesh.nents(i), -1);
      numberValidEntries(mesh.nents(i), buf_ents[i], ent_ids[i]);
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

    //****************Convert Tags to the picpart***********
    Omega_h::Write<Omega_h::LO> new_safe(picpart->nelems(), 0);
    Omega_h::Write<Omega_h::GO> new_elem_gid(picpart->nelems(), 0);
    Omega_h::Write<Omega_h::LO> new_ent_owners(picpart->nelems(), 0);
    const auto convertArraysToPicpart = OMEGA_H_LAMBDA(Omega_h::LO elem_id) {
      const Omega_h::LO new_elem = ent_ids[dim][elem_id];
      //TODO remove this conditional using padding?
      if (new_elem >= 0) {
        new_safe[new_elem] = is_safe[elem_id];
        new_elem_gid[new_elem] = elem_gid[elem_id];
        new_ent_owners[new_elem] = owner[elem_id];
      }
    };
    Omega_h::parallel_for(mesh.nelems(), convertArraysToPicpart, "convertArraysToPicpart");
    global_ids_per_dim[dim] = Omega_h::Read<Omega_h::GO>(new_elem_gid);
    is_ent_safe = Omega_h::Read<Omega_h::LO>(new_safe);

    //**************** Build communication information ********************//
    //TODO create communication information for each entity dimension
    picpart->set_comm(lib->world());
    Omega_h::Write<Omega_h::LO> picpart_offset_nelms(comm_size+1,0);
    calculateOwnerOffset(new_ent_owners, picpart_offset_nelms);
    setupComm(3, rank_offset_nelms, picpart_offset_nelms,
              new_ent_owners);
  }
}

namespace {

  void calculateOwnerOffset(Omega_h::Write<Omega_h::LO>& owner,
                            Omega_h::Write<Omega_h::LO>& offset_nents) {
    const int comm_size = offset_nents.size()-1;
    //TODO put this on device
    for (int i = 0; i < owner.size(); ++i) {
      ++offset_nents[owner[i]+1];
    }
    //Exclusive sum over elems per rank
    for (int i = 1; i <= comm_size; ++i) {
      offset_nents[i] += offset_nents[i-1];
    }
  }

  void createGlobalNumbering(Omega_h::Write<Omega_h::LO>& owner,
                             Omega_h::Write<Omega_h::LO>& rank_offset_nelms,
                             Omega_h::Write<Omega_h::GO>& elem_gid) {
    const int comm_size = rank_offset_nelms.size();
    calculateOwnerOffset(owner,rank_offset_nelms);

    //TODO Put this on device
    //Globally number the elements
    int* elem_gid_rank = new int[comm_size];
    for (int i = 0; i < comm_size; ++i)
      elem_gid_rank[i] = rank_offset_nelms[i];
    for (int i = 0; i < owner.size(); ++i) {
      elem_gid[i] = elem_gid_rank[owner[i]]++;
    }
  }

  void bfsBufferLayers(Omega_h::Mesh& mesh, int bridge_dim, int safe_layers, int ghost_layers,
                       Omega_h::Write<Omega_h::LO>& is_safe, Omega_h::Write<Omega_h::LO>& is_visited,
                       Omega_h::Write<Omega_h::LO>& owner, Omega_h::Write<Omega_h::LO>& has_part) {
    //runBFS to calculate safe zone and buffered region
    Omega_h::Write<Omega_h::LO> is_visited_next(mesh.nelems());
    const auto initVisit = OMEGA_H_LAMBDA( Omega_h::LO elem_id){
      is_visited_next[elem_id] = is_visited[elem_id];
    };
    Omega_h::parallel_for(mesh.nelems(), initVisit, "initVisit");

    const auto bridge2elems = mesh.ask_up(bridge_dim, mesh.dim());
    auto ghostingBFS = OMEGA_H_LAMBDA( Omega_h::LO bridge_id) {
      const auto deg = bridge2elems.a2ab[bridge_id + 1] - bridge2elems.a2ab[bridge_id];
      const auto firstElm = bridge2elems.a2ab[bridge_id];
      bool is_visited_here = false;
      for (int j = 0; j < deg; ++j) {
        const auto elm = bridge2elems.ab2b[firstElm+j];
        is_visited_here |= is_visited[elm];
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
      has_part[owner[elm_id]] |= is_visited[elm_id];
    };
    for (int i = 0; i < ghost_layers; ++i) {
      Omega_h::parallel_for(mesh.nents(bridge_dim), ghostingBFS,"ghostingBFS");
      if (i < safe_layers)
        Omega_h::parallel_for(mesh.nelems(), copySafe, "copySafe");
      Omega_h::parallel_for(mesh.nelems(), copyVisit, "copyVisit");
    }
  }

  void setSafeEnts(Omega_h::Mesh& mesh, int dim, int size, Omega_h::Write<Omega_h::LO>& has_part, 
                   Omega_h::Write<Omega_h::LO>& owner, Omega_h::Write<Omega_h::LO>& buf) {
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
          buf[ent] |= is_buffered;
        }
      };
      Omega_h::parallel_for(size, setSafeEnts, "setSafeEnts");
    }
  }

  //TODO Replace with omega_h reduce/scan
  Omega_h::LO sumPositives(Omega_h::LO size, Omega_h::Write<Omega_h::LO>& arr) {
    Omega_h::LO sum = 0;
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    Kokkos::parallel_reduce(size, KOKKOS_LAMBDA(const int i, Omega_h::LO& lsum) {
        lsum += arr[i] > 0 ;
      }, sum);
#endif
    return sum;
  }
  void numberValidEntries(Omega_h::LO size, Omega_h::Write<Omega_h::LO>& is_valid, 
                          Omega_h::Write<Omega_h::LO>& numbering) {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    Kokkos::parallel_scan(size, KOKKOS_LAMBDA(const int& i, int& num, const bool& final) {
      num += is_valid[i];
      numbering[i] += num * final * is_valid[i];
    });
#endif
  }

  void gatherCoords(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO>& vert_ids,
                    Omega_h::Write<Omega_h::Real>& new_coords) {
    Omega_h::Reals n2c = mesh.coords();
    auto gatherCoordsL = OMEGA_H_LAMBDA(Omega_h::LO vert_id)  {
      const Omega_h::LO first_vert = vert_id * 3;
      const Omega_h::LO first_new_vert = vert_ids[vert_id]*3;
      const int deg = (first_new_vert >=0) * mesh.dim();
      for (int i = 0; i < deg; ++i)
        new_coords[first_new_vert + i] = n2c[first_vert + i];
    };
    Omega_h::parallel_for(mesh.nverts(), gatherCoordsL, "gatherCoords");
  }

  void buildAndClassify(Omega_h::Mesh& full_mesh, Omega_h::Mesh* picpart, int dim, int num_ents,
                        Omega_h::Write<Omega_h::LO>& ent_ids, Omega_h::Write<Omega_h::LO>& vert_ids,
                        Omega_h::Write<Omega_h::Real>& new_coords) {
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
