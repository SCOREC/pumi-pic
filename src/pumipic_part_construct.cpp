#include "pumipic_part_construct.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <Omega_h_tag.hpp>
#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_scalar.hpp> //divide
#include <Omega_h_mark.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_compare.hpp>
#include <Omega_h_reduce.hpp>
#include <Omega_h_int_scan.hpp> //exclusive scan
#include <Omega_h_scan.hpp> //parallel_scan

namespace {
  void createGlobalNumbering(Omega_h::Write<Omega_h::LO>& owner,
                             Omega_h::Write<Omega_h::LO>& rank_offset_nelms,
                             Omega_h::Write<Omega_h::LO>& elem_gid);
  void bfsBufferLayers(Omega_h::Mesh& mesh, int bridge_dim, int safe_layers, int ghost_layers,
                       Omega_h::Write<Omega_h::LO>& is_safe, Omega_h::Write<Omega_h::LO>& is_visited,
                       Omega_h::Write<Omega_h::LO>& owner, Omega_h::Write<Omega_h::LO>& has_part);
  void setSafeEnts(Omega_h::Mesh& mesh, int dim, int size, Omega_h::Write<Omega_h::LO>& has_part, 
                   Omega_h::Write<Omega_h::LO>& owner, Omega_h::Write<Omega_h::LO>& buf);
  Omega_h::LO sumArray(Omega_h::LO size, Omega_h::Write<Omega_h::LO>& arr);
  Omega_h::LOs numberValidEntries(Omega_h::LO size, Omega_h::Write<Omega_h::LO> is_valid);
  void gatherCoords(Omega_h::Mesh& mesh, Omega_h::LOs vert_ids,
                    Omega_h::Write<Omega_h::Real>& new_coords);
  void buildAndClassify(Omega_h::Mesh& full_mesh, Omega_h::Mesh* picpart, int dim, int num_ents,
                        Omega_h::LOs ent_ids, Omega_h::LOs vert_ids,
                        Omega_h::Write<Omega_h::Real>& new_coords);
}

namespace pumipic {
  void constructPICParts(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO>& owner,
                         int safe_layers, int ghost_layers, Omega_h::Mesh* picpart, int debug) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (ghost_layers < safe_layers) {
      if (!rank)
        fprintf(stderr, "Ghost layers must be >= safe layers");
      throw 1;
    }

    int dim = mesh.dim();
    int ne = mesh.nelems();
    /************* Globally Number Element **********/
    Omega_h::Write<Omega_h::LO> elem_gid(ne);
    Omega_h::Write<Omega_h::LO> rank_offset_nelms(comm_size+1,0);
    createGlobalNumbering(owner, rank_offset_nelms, elem_gid);

    // **********Determine safe zone and ghost region**************** //
    Omega_h::Write<Omega_h::LO> is_safe(ne);
    Omega_h::Write<Omega_h::LO> is_visited(ne);
    const auto initVisit = OMEGA_H_LAMBDA( Omega_h::LO elem_id){
      is_visited[elem_id] = is_safe[elem_id] = (owner[elem_id] == rank);
    };
    Omega_h::parallel_for(ne, initVisit, "initVisit");
    Omega_h::Write<Omega_h::LO> has_part(comm_size);
    Omega_h::parallel_for(comm_size, OMEGA_H_LAMBDA(Omega_h::LO i) {
      has_part[i] = (i == rank);
    });
    int bridge_dim = 0;
    bfsBufferLayers(mesh, bridge_dim, safe_layers, ghost_layers, is_safe, is_visited, 
                    owner, has_part);

    if (debug >= 2) {
      Omega_h::parallel_for(comm_size, OMEGA_H_LAMBDA(Omega_h::LO i) {
        if (has_part[i])
          printf("Rank %d is keeping %d\n", rank, i);
      });
    }
    if (debug >= 3) {
      //*************Render the 0th rank after BFS*************//
      mesh.add_tag(dim, "global_id", 1, Omega_h::Read<Omega_h::LO>(elem_gid));
      mesh.add_tag(dim, "safe", 1, Omega_h::Read<int>(is_safe));
      mesh.add_tag(dim, "visited", 1, Omega_h::Read<int>(is_visited));
      if (rank == 0) {
        Omega_h::vtk::write_parallel("rendered", &mesh, dim);
      }
    }

    /***************** Count Number of Entities in the PICpart *************/    
    //Mark all entities owned by a part with has_part[part] = true as staying
    Omega_h::Write<Omega_h::LO>** buf_ents = new Omega_h::Write<Omega_h::LO>*[dim+1];
    for (int i = 0; i <= dim; ++i) 
      buf_ents[i] = new Omega_h::Write<Omega_h::LO>(mesh.nents(i),0);
    for (int i = 0; i <= dim; ++i)
      setSafeEnts(mesh, i, ne, has_part, owner, *(buf_ents[i]));
  
    //Gather number of entities remaining in the picpart
    Omega_h::GO* num_ents = new Omega_h::GO[dim+1];
    for (int i = 0; i <= dim; ++i)
      num_ents[i] = sumArray(mesh.nents(i), *(buf_ents[i]));
    if (debug >= 1)
      printf("Rank %d has <v e f r> %ld %ld %ld %ld\n", rank, 
             num_ents[0], num_ents[1], num_ents[2], num_ents[3]);

    /**************** Create numberings for the entities on the picpart **************/
    Omega_h::LOs ent_ids[4];
    for (int i = 0; i <= dim; ++i) {
      //Default the value to the number of entities in the pic part (for padding)
      auto num = numberValidEntries(mesh.nents(i), *(buf_ents[i]));
      ent_ids[i] = num;
    }

    //************Build a new mesh as the picpart**************
    //Gather coordinates
    //Pad the array by 1 set of coordinates
    Omega_h::Write<Omega_h::Real> new_coords((num_ents[0])*dim,0);
    gatherCoords(mesh, ent_ids[0], new_coords);

    for (int i = dim; i >= 0; --i)
      buildAndClassify(mesh,picpart,i,num_ents[i], ent_ids[i], ent_ids[0], new_coords);
    Omega_h::finalize_classification(picpart);

    delete [] num_ents;

    //****************Convert Tags to the picpart***********
    Omega_h::Write<Omega_h::LO> new_safe(picpart->nelems(), 0);
    Omega_h::Write<Omega_h::LO> new_elem_gid(picpart->nelems(), 0);
    Omega_h::LOs new_elem_ids = ent_ids[dim];
    Omega_h::Write<Omega_h::LO> new_ent_owners(picpart->nelems(), 0);
    const auto convertArraysToPicpart = OMEGA_H_LAMBDA(Omega_h::LO elem_id) {
      const Omega_h::LO new_elem = new_elem_ids[elem_id];
      //TODO remove this conditional using padding?
      if (new_elem >= 0) {
        new_safe[new_elem] = is_safe[elem_id];
        new_elem_gid[new_elem] = elem_gid[elem_id];
        new_ent_owners[new_elem] = owner[elem_id];
      }
    };
    Omega_h::parallel_for(mesh.nelems(), convertArraysToPicpart, "convertArraysToPicpart");
    picpart->add_tag(dim, "global_id", 1, Omega_h::Read<Omega_h::LO>(new_elem_gid));
    picpart->add_tag(dim, "safe", 1, Omega_h::Read<Omega_h::LO>(new_safe));

    //Cleanup
    for (int i = 0; i <= dim; ++i) {
      delete buf_ents[i];
    }
    delete [] buf_ents;
  }
}

namespace {

  void createGlobalNumbering(Omega_h::Write<Omega_h::LO>& owner,
                             Omega_h::Write<Omega_h::LO>& rank_offset_nelms,
                             Omega_h::Write<Omega_h::LO>& elem_gid) {
    const int comm_size = rank_offset_nelms.size();
    //TODO put this on device
    Omega_h::parallel_for(owner.size(), OMEGA_H_LAMBDA(Omega_h::LO i) {
      const auto owner_idx = owner[i]+1;
      Kokkos::atomic_fetch_add(&rank_offset_nelms[owner_idx],1);
    });
   
    //Exclusive sum over elems per rank
    Omega_h::Read<Omega_h::LO> rank_offset_nelms_r(rank_offset_nelms);
    auto rank_offsets = Omega_h::offset_scan(rank_offset_nelms_r);
    //Globally number the elements
    Omega_h::Write<Omega_h::LO> elem_gid_rank(comm_size,-1);
    Omega_h::parallel_for(comm_size, OMEGA_H_LAMBDA(Omega_h::LO i) {
      elem_gid_rank[i] = rank_offsets[i];
    });
    Omega_h::parallel_for(owner.size(), OMEGA_H_LAMBDA(Omega_h::LO i) {
      const auto elm_owner = owner[i];
      const auto elm_rank = elem_gid_rank[elm_owner];
      elem_gid[i] = elm_rank;
      elem_gid_rank[elm_owner] = elm_rank+1;
    });
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

  Omega_h::LO sumArray(Omega_h::LO size, Omega_h::Write<Omega_h::LO>& arr) {
    Omega_h::LO sum = 0;
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    Kokkos::parallel_reduce(size, OMEGA_H_LAMBDA(const int i, Omega_h::LO& lsum) {
        lsum += arr[i] > 0 ;
      }, sum);
#endif
    return sum;
  }

  Omega_h::LOs numberValidEntries(Omega_h::LO size, Omega_h::Write<Omega_h::LO> is_valid) {
    Omega_h::LOs is_valid_r(is_valid);
    return Omega_h::offset_scan(is_valid_r);
  }

  void gatherCoords(Omega_h::Mesh& mesh, Omega_h::LOs vert_ids,
                    Omega_h::Write<Omega_h::Real>& new_coords) {
    const auto meshDim = mesh.dim();
    Omega_h::Reals n2c = mesh.coords();
    auto gatherCoordsL = OMEGA_H_LAMBDA(Omega_h::LO vert_id)  {
      const Omega_h::LO first_vert = vert_id * 3;
      const Omega_h::LO first_new_vert = vert_ids[vert_id]*3;
      const int deg = (first_new_vert >=0 ) * meshDim;
      for (int i = 0; i < deg; ++i)
        new_coords[first_new_vert + i] = n2c[first_vert + i];
    };
    Omega_h::parallel_for(mesh.nverts(), gatherCoordsL, "gatherCoords");
  }

  void buildAndClassify(Omega_h::Mesh& full_mesh, Omega_h::Mesh* picpart, int dim, int num_ents,
                        Omega_h::LOs ent_ids, Omega_h::LOs vert_ids,
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
