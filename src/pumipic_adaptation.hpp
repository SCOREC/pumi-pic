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

  virtual void refine(Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges,
      LOs keys2midverts, Int prod_dim, LOs keys2prods, LOs prods2new_ents,
      LOs same_ents2old_ents, LOs same_ents2new_ents) {
    int mesh_dim = old_mesh.dim();
    if (prod_dim != mesh_dim) return;

    Write<LO> elem_dim(old_mesh.nelems());
    Write<LO> elem2new(old_mesh.nelems());

    auto old_adj = old_mesh.ask_up(EDGE, mesh_dim);
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

    parallel_for(same_ents2old_ents.size(), OMEGA_H_LAMBDA(LO i) {
      LO oldElem = same_ents2old_ents[i];
      elem_dim[oldElem] = mesh_dim;
      elem2new[oldElem] = same_ents2new_ents[i];
    });

    ps::kkLidView ptclElems_cpy = ptclElems;
    auto ptclPos = ptcls->get<0>();
    auto cells2nodes = new_mesh.get_adj(dim, Omega_h::VERT).ab2b;
    auto nodes2coords = new_mesh.coords();
    Kokkos::parallel_for(ptclElems_cpy.size(), KOKKOS_LAMBDA(const int ptcl) {
      auto oldElem = ptclElems_cpy[ptcl];
      if (elem_dim[oldElem] == mesh_dim)
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
    printf("==CoarsenFound==\n");
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