#ifndef PUMIPIC_ADAPTATION_HPP
#define PUMIPIC_ADAPTATION_HPP

#include "pumipic_kktypes.hpp"
#include "Omega_h_align.hpp"
#include "Omega_h_scalar.hpp"
#include "Omega_h_element.hpp"
#include <MemberTypeLibraries.h>

using particle_structs::MemberTypes;
//position, elem, dim
typedef MemberTypes<double[3], int, int, int> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef ps::ParticleStructure<Type,ExeSpace> PS;

namespace Omega_h {
  template<int mesh_dim>
  struct ParticleAdapt : public UserTransfer {

  struct ModifiedElem {
    LO key=-1;
    LO offset=-1;
    LO rotation=-1;
  };

  PS* ptcls;

  ParticleAdapt(PS* particles) {
    ptcls = particles;
  }

  Write<LO> getUnchanged(Mesh& old_mesh, Int dim, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    Write<LO> old2New(old_mesh.nents(dim), -1);
    parallel_for(same_ents2old_ents.size(), OMEGA_H_LAMBDA(LO i) {
      LO oldElem = same_ents2old_ents[i];
      old2New[oldElem] = same_ents2new_ents[i];
    });
    return old2New;
  }

  //TODO: test refinment multiple times
  virtual void refine(Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges,
      LOs keys2midverts, Int prod_dim, LOs keys2prods, LOs prods2new_ents,
      LOs same_ents2old_ents, LOs same_ents2new_ents) {
    
    if (prod_dim != mesh_dim) return;
    Write<LO> same_old2New = getUnchanged(old_mesh, prod_dim, same_ents2old_ents, same_ents2new_ents);

    Kokkos::View<ModifiedElem*> modified("modified_elems", old_mesh.nelems());
    auto old_edge2Elem = old_mesh.ask_up(EDGE, mesh_dim);

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
    auto ptclElem = ptcls->get<1>();
    auto ptclDim = ptcls->get<2>();
    auto new_verts2coords = new_mesh.coords();
    auto old_verts2coords = old_mesh.coords();
    auto old_elem2verts = old_mesh.get_adj(mesh_dim, VERT).ab2b;
    auto new_elem2edge = new_mesh.get_adj(mesh_dim, EDGE);
    auto nEdges = element_degree(new_mesh.family(), mesh_dim, EDGE);

    //Update modified elements
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto oldElem = ptclElem(pid);
      if (same_old2New[oldElem] != -1)
        ptclElem(pid) = same_old2New[oldElem];
      else if (modified[oldElem].offset != -1) {
        auto key = modified[oldElem].key;

        Vector<mesh_dim> pos;
        for (int i = 0; i<mesh_dim; i++)
          pos[i] = ptclPos(pid,i);

        // auto splitPos = get_vector<mesh_dim>(new_verts2coords, LO(keys2midverts[key]));
        auto oldVerts = gather_verts<mesh_dim+1>(old_elem2verts, LO(oldElem));
        auto oldCoords = gather_vectors<mesh_dim+1,mesh_dim>(old_verts2coords, oldVerts);
        auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(pos, oldCoords);
        int side = baryCoords[0] > baryCoords[2] ? 0 : 2;
        if (are_close(baryCoords[0], baryCoords[2])) side = 1;
        const int rotateDirCode[3][2] = {{0,1},{0,0},{1,0}};
        int rotated = rotateDirCode[side][modified[oldElem].rotation];
        auto prod = keys2prods[key] + modified[oldElem].offset*2 + rotated;
        ptclElem(pid) = prods2new_ents[prod];
        ptclDim(pid) = mesh_dim;
        if (side == 1) {
          auto edgeIdx = ptclElem(pid)*nEdges + modified[oldElem].rotation + side;
          ptclElem(pid) = new_elem2edge.ab2b[edgeIdx];
          ptclDim(pid) = 1;
        }
      }
      else {
        printf("WARNING: element skipped during particle adaptation\n");
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