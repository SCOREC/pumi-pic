#ifndef PUMIPIC_ADAPTATION_HPP
#define PUMIPIC_ADAPTATION_HPP

#include "pumipic_kktypes.hpp"

using particle_structs::MemberTypes;
typedef MemberTypes<double[3], int> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef ps::ParticleStructure<Type,ExeSpace> PS;

namespace Omega_h {
  template<int dim>
  struct ParticleAdapt : public UserTransfer {

  PS* ptcls;
  ps::kkLidView ptclElems; //TODO: might require reset after adaptation complete

  ps::kkLidView getPtcls() {
    ps::kkLidView particleElems("ptcl_elems", ptcls->nPtcls());
    auto copyPtclsPerElem = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if(mask > 0) particleElems(pid) = e;
    };
    ps::parallel_for(ptcls, copyPtclsPerElem);
    return particleElems;
  }

  ParticleAdapt(PS* particles) {
    ptcls = particles;
    ptclElems = getPtcls();
  }

  void getUpdatedEntities(Write<LO>& elem_dim, Write<LO>& elem2new, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    parallel_for(same_ents2old_ents.size(), OMEGA_H_LAMBDA(LO i) {
      LO oldElem = same_ents2old_ents[i];
      elem_dim[oldElem] = dim;
      elem2new[oldElem] = same_ents2new_ents[i];
      printf("RENAME %d TO %d\n", oldElem, same_ents2new_ents[i]);
    });
  }

  virtual void refine(Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges,
      LOs keys2midverts, Int prod_dim, LOs keys2prods, LOs prods2new_ents,
      LOs same_ents2old_ents, LOs same_ents2new_ents) { //TODO: test refinment multiple times
    if (prod_dim != dim) return;

    Write<LO> elem_dim(old_mesh.nelems(), -1);
    Write<LO> elem2new(old_mesh.nelems(), -1);

    getUpdatedEntities(elem_dim, elem2new, same_ents2old_ents, same_ents2new_ents);

    auto old_adj = old_mesh.ask_up(EDGE, dim);
    parallel_for(keys2edges.size(), OMEGA_H_LAMBDA(LO key) {
      LO edge = keys2edges[key];
      auto elem_begin = old_adj.a2ab[edge];
      auto elem_end = old_adj.a2ab[edge + 1];
      for (auto idx = elem_begin; idx < elem_end; ++idx) {
        auto elem = old_adj.ab2b[idx];
        elem_dim[elem] = 0;
        elem2new[elem] = key;
      }
    });

    ps::kkLidView ptclElems_cpy = ptclElems;
    auto ptclPos = ptcls->get<0>();
    auto cells2nodes = new_mesh.get_adj(dim, Omega_h::VERT).ab2b;
    auto nodes2coords = new_mesh.coords();
    Kokkos::parallel_for(ptclElems_cpy.size(), KOKKOS_LAMBDA(const int ptcl) {
      auto oldElem = ptclElems_cpy[ptcl];
      if (elem_dim[oldElem] == dim)
        ptclElems_cpy[ptcl] = elem2new[oldElem];
      else {
        auto key = elem2new[oldElem];

        Vector<dim> pos;
        for (int i = 0; i<dim; i++)
          pos[i] = ptclPos(ptcl,i);

        for (auto prod=keys2prods[key]; prod < keys2prods[key+1]; prod++){//TODO: update to only search two elem
          auto entity = prods2new_ents[prod];
          auto elmVerts = Omega_h::gather_verts<dim+1>(cells2nodes, Omega_h::LO(entity));
          auto vtxCoords = Omega_h::gather_vectors<dim+1,dim>(nodes2coords, elmVerts);
          auto xi = barycentric_from_global<dim,dim>(pos, vtxCoords);
          if (is_barycentric_inside(xi, .0000001)){
            ptclElems_cpy[ptcl] = entity;
            return;
          }
        }
        printf("[ERROR]: Particle not inside any elem\n"); //TODO: change just set to last element
      }
    });
  }
  virtual void coarsen(Mesh& old_mesh, Mesh& new_mesh, LOs keys2verts,
      Adj keys2doms, Int prod_dim, LOs prods2new_ents, LOs same_ents2old_ents,
      LOs same_ents2new_ents) {
    if (prod_dim != dim) return;
    printf("==CoarsenFound==\n");

    printf("VERTS %d NEW %d SAME %d\n", keys2verts.size(), prods2new_ents.size(), same_ents2old_ents.size());

    Write<LO> elem_dim(old_mesh.nelems(), -1);
    Write<LO> elem2new(old_mesh.nelems(), -1);

    getUpdatedEntities(elem_dim, elem2new, same_ents2old_ents, same_ents2new_ents);

    parallel_for(keys2verts.size(), OMEGA_H_LAMBDA(LO key) {
      auto elem_begin = keys2doms.a2ab[key];
      auto elem_end = keys2doms.a2ab[key + 1];
      for (auto idx = elem_begin; idx < elem_end; ++idx) {
        auto elem = keys2doms.ab2b[idx];
        elem_dim[elem] = dim;
        elem2new[elem] = prods2new_ents[idx];
      }
    });

    printf("==ADAPTATION RESULTS==\n");
    parallel_for(elem2new.size(), OMEGA_H_LAMBDA(LO key) {
      printf("OLD %d NEW %d FOUND %d\n", key, elem2new[key], elem_dim[key]);
    });

    ps::kkLidView ptclElems_cpy = ptclElems;
    Kokkos::parallel_for(ptclElems_cpy.size(), KOKKOS_LAMBDA(const int ptcl) {
      auto oldElem = ptclElems_cpy[ptcl];
      if (elem_dim[oldElem] == dim)
        ptclElems_cpy[ptcl] = elem2new[oldElem];
    });

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