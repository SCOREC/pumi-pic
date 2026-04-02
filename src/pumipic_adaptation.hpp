#ifndef PUMIPIC_ADAPTATION_HPP
#define PUMIPIC_ADAPTATION_HPP

#include "pumipic_kktypes.hpp"
#include "Omega_h_align.hpp"
#include "Omega_h_scalar.hpp"

using particle_structs::MemberTypes;
typedef MemberTypes<double[3], int> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef ps::ParticleStructure<Type,ExeSpace> PS;

namespace Omega_h {
  template<int mesh_dim>
  struct ParticleAdapt : public UserTransfer {

  struct PtclInfo {
    int elem;
    int dim;
  };

  struct ModifiedElem {
    LO offset=-1;
    LO key=-1;
    LO rotation=-1;
  };

  PS* ptcls;
  Kokkos::View<PtclInfo*> ptclElems; //TODO: might require reset after adaptation complete
  Adj old_vtx2Elem;
  Adj old_edge2Elem;
  Adj old_face2Elem;

  ParticleAdapt(PS* particles) {
    init(particles);
  }

  void init(PS* particles) {
    ptcls = particles;
    ptclElems = Kokkos::View<PtclInfo*>("ptcl_info", ptcls->nPtcls());
    auto copyPtclsPerElem = KOKKOS_CLASS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if(mask > 0) ptclElems(pid) = PtclInfo(e, mesh_dim);
    };
    ps::parallel_for(ptcls, copyPtclsPerElem);
  }

  void updateAdjacency(Mesh& old_mesh) {
    old_vtx2Elem = old_mesh.ask_up(VERT, mesh_dim);
    old_edge2Elem = old_mesh.ask_up(EDGE, mesh_dim);
    old_face2Elem = old_mesh.ask_up(FACE, mesh_dim);
  }

  KOKKOS_INLINE_FUNCTION
  LO getParentElement(LO ptclID) const {
    auto ptcl = ptclElems(ptclID);
    if (ptcl.dim == mesh_dim)
      return ptcl.elem;
    else if (ptcl.dim == 0) {
      auto firstElem = old_vtx2Elem.a2ab[ptcl.elem];
      return old_vtx2Elem.ab2b[firstElem];
    }
    else if (ptcl.dim == 1) {
      auto firstElem = old_edge2Elem.a2ab[ptcl.elem];
      return old_edge2Elem.ab2b[firstElem];
    }
    else if (ptcl.dim == 2) {
      auto firstElem = old_face2Elem.a2ab[ptcl.elem];
      return old_face2Elem.ab2b[firstElem];
    }
    else {
      printf("ERROR: PARTICLE ELEM NOT FOUND\n");
      return 0;
    }
  }

  void updateEntities(Mesh& old_mesh, Int dim, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    Write<LO> old2New(old_mesh.nents(dim), -1);
    parallel_for(same_ents2old_ents.size(), OMEGA_H_LAMBDA(LO i) {
      LO oldElem = same_ents2old_ents[i];
      old2New[oldElem] = same_ents2new_ents[i];
    });

    Kokkos::parallel_for(ptclElems.size(), KOKKOS_CLASS_LAMBDA(const int id) {
      auto oldPtcl = ptclElems(id);
      if (oldPtcl.dim == dim && old2New[oldPtcl.elem] != -1)
        ptclElems[id].elem = old2New[oldPtcl.elem];
    });
  }

  //TODO: test refinment multiple times
  virtual void refine(Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges,
      LOs keys2midverts, Int prod_dim, LOs keys2prods, LOs prods2new_ents,
      LOs same_ents2old_ents, LOs same_ents2new_ents) {
    
    updateEntities(old_mesh, prod_dim, same_ents2old_ents, same_ents2new_ents);
    if (prod_dim != mesh_dim) return;
    updateAdjacency(old_mesh);

    Kokkos::View<ModifiedElem*> modified("updated_elems", old_mesh.nelems());

    //Gather modified elements
    parallel_for(keys2edges.size(), KOKKOS_CLASS_LAMBDA(LO key) {
      LO edge = keys2edges[key];
      auto elem_begin = old_edge2Elem.a2ab[edge];
      auto elem_end = old_edge2Elem.a2ab[edge + 1];
      for (auto idx = elem_begin; idx < elem_end; ++idx) {
        auto elem = old_edge2Elem.ab2b[idx];
        modified[elem].key = key;
        modified[elem].offset = idx-elem_begin;
        modified[elem].rotation = code_rotation(old_edge2Elem.codes[idx]);
      }
    });

    auto ptclPos = ptcls->get<0>();
    auto new_verts2coords = new_mesh.coords();
    auto old_elem2verts = old_mesh.get_adj(mesh_dim, VERT).ab2b;
    auto old_verts2coords = old_mesh.coords();

    Kokkos::parallel_for(ptclElems.size(), KOKKOS_CLASS_LAMBDA(const int ptcl) {
      auto oldElem = getParentElement(ptcl);
      if (modified[oldElem].offset != -1) {
        auto key = modified[oldElem].key;

        Vector<mesh_dim> pos;
        for (int i = 0; i<mesh_dim; i++)
          pos[i] = ptclPos(ptcl,i);

        auto splitPos = get_vector<mesh_dim>(new_verts2coords, LO(keys2midverts[key]));
        auto oldVerts = gather_verts<mesh_dim+1>(old_elem2verts, LO(oldElem));
        auto oldCoords = gather_vectors<mesh_dim+1,mesh_dim>(old_verts2coords, oldVerts);
        auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(pos, oldCoords);
        int side = baryCoords[0] > baryCoords[2] ? 0 : 2;
        if (are_close(baryCoords[0], baryCoords[2])) side = 1;
        const int rotateDirCode[3][2] = {{0,1},{0,0},{1,0}};
        int rotated = rotateDirCode[side][modified[oldElem].rotation];
        auto prod = keys2prods[key] + modified[oldElem].offset*2 + rotated;
        ptclElems[ptcl].elem = prods2new_ents[prod];
      }
    });
  }
  virtual void coarsen(Mesh& old_mesh, Mesh& new_mesh, LOs keys2verts,
      Adj keys2doms, Int prod_dim, LOs prods2new_ents, LOs same_ents2old_ents,
      LOs same_ents2new_ents) {
    if (prod_dim != mesh_dim) return;
    printf("==CoarsenFound==\n");

    // printf("VERTS %d NEW %d SAME %d\n", keys2verts.size(), prods2new_ents.size(), same_ents2old_ents.size());

    // Write<LO> elem_dim(old_mesh.nelems(), -1);
    // Write<LO> elem2new(old_mesh.nelems(), -1);

    // getUpdatedEntities(elem_dim, elem2new, same_ents2old_ents, same_ents2new_ents);

    // parallel_for(keys2verts.size(), OMEGA_H_LAMBDA(LO key) {
    //   auto elem_begin = keys2doms.a2ab[key];
    //   auto elem_end = keys2doms.a2ab[key + 1];
    //   for (auto idx = elem_begin; idx < elem_end; ++idx) {
    //     auto elem = keys2doms.ab2b[idx];
    //     elem_dim[elem] = mesh_dim;
    //     elem2new[elem] = prods2new_ents[idx];
    //   }
    // });

    // printf("==COARSEN RESULTS==\n");
    // parallel_for(elem2new.size(), OMEGA_H_LAMBDA(LO key) {
    //   printf("OLD %d NEW %d FOUND %d\n", key, elem2new[key], elem_dim[key]);
    // });

    // Kokkos::View<PtclInfo*> ptclElems_cpy = ptclElems;
    // Kokkos::parallel_for(ptclElems_cpy.size(), KOKKOS_LAMBDA(const int ptcl) {
      // auto oldElem = ptclElems_cpy[ptcl];
      // if (elem_dim[oldElem] == mesh_dim)
      //   ptclElems_cpy[ptcl] = elem2new[oldElem];
    // });

    };
  virtual void swap(Mesh& old_mesh, Mesh& new_mesh, Int prod_dim,
      LOs keys2edges, LOs keys2prods, LOs prods2new_ents,
      LOs same_ents2old_ents, LOs same_ents2new_ents) {
    printf("==SwapFound==\n");
    };
  virtual void swap_copy_verts(Mesh& old_mesh, Mesh& new_mesh) {
    printf("==SwapCopyVertsFound==\n");
  };
};

}//namespace
#endif //define