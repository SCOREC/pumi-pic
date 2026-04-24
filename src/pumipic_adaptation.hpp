#ifndef PUMIPIC_ADAPTATION_HPP
#define PUMIPIC_ADAPTATION_HPP

#include "pumipic_kktypes.hpp"
#include "Omega_h_align.hpp"
#include "Omega_h_scalar.hpp"
#include "Omega_h_element.hpp"
#include <MemberTypeLibraries.h>

using particle_structs::MemberTypes;
enum MemberIndex{POS, PARENT, CHILD, DIM, PID};
typedef MemberTypes<double[3], Omega_h::LO, int, int, int> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef ps::ParticleStructure<Type,ExeSpace> PS;

namespace Omega_h {
  template<int mesh_dim>
  struct ParticleAdapt : public UserTransfer {

  struct ModifiedElem {
    LO key=-1;
    LO offset=-1;
    LO code=-1;
  };

  PS*& ptcls;
  Mesh& mesh;
  LOs old2New[mesh_dim+1];
  Adj new_upward[mesh_dim];
  Adj new_downward[mesh_dim];
  PS::Slice<POS> pPos;
  PS::Slice<PARENT> pParent;
  PS::Slice<CHILD> pChild;
  PS::Slice<DIM> pDim;


  ParticleAdapt(PS*& ptclsIn, Mesh& meshIn) : ptcls(ptclsIn), mesh(meshIn) {
    update(meshIn);
  }

  void update(Mesh& meshIn) {
    pPos = ptcls->get<POS>();
    pParent = ptcls->get<PARENT>();
    pChild = ptcls->get<CHILD>();
    pDim = ptcls->get<DIM>();
    for (int i=0; i<mesh_dim; i++) {
      new_upward[i] = meshIn.ask_up(i, mesh_dim);
      new_downward[i] = meshIn.ask_down(mesh_dim, i);
    }
  }

  KOKKOS_INLINE_FUNCTION
  int getChildIndex(int dim, LO parent, LO child) const {
    auto degree = simplex_degree(mesh_dim, dim);
    for (auto i = 0; i < 3; i++)
      if (new_downward[dim].ab2b[parent*degree + i] == child) return i;
    return 0;
  }

  //TODO: replace previous with this
  KOKKOS_INLINE_FUNCTION
  void setPtcl(LO pid, int dim, LO parent, LO child) const {
    auto degree = simplex_degree(mesh_dim, dim);
    int childIdx = -1;
    for (auto i = 0; i < degree; i++)
      if (new_downward[dim].ab2b[parent*degree + i] == child) childIdx = i;

    pDim(pid) = dim;
    pParent(pid) = parent;
    pChild(pid) = childIdx;
  }

  KOKKOS_INLINE_FUNCTION
  LO getChildElem(LO pid) const {
    if (pDim(pid) == mesh_dim) return pParent(pid);
    auto degree = simplex_degree(mesh_dim, pDim(pid));
    return new_downward[pDim(pid)].ab2b[pParent(pid)*degree + pChild(pid)];
  }

  KOKKOS_INLINE_FUNCTION
  LO getChildElem(const Adj downward[mesh_dim], int dim, LO parent, int index) const {
    if (dim == mesh_dim) return parent;
    auto degree = simplex_degree(mesh_dim, dim);
    return downward[dim].ab2b[parent*degree + index];
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
    
    old2New[prod_dim] = getUnchanged(old_mesh, prod_dim, same_ents2old_ents, same_ents2new_ents);
    if (prod_dim != mesh_dim) return;

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
        modified[elem].code = old_edge2Elem.codes[idx];
      }
    });

    update(new_mesh);
    auto old_verts2coords = old_mesh.coords();
    Adj old_downward[mesh_dim];
    for (int i=0; i<mesh_dim; i++)
      old_downward[i] = old_mesh.ask_down(mesh_dim, i);

    //Update modified elements
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto oldElem = pParent(pid);
      if (old2New[mesh_dim][oldElem] != -1) //update unchanged element id
        pParent(pid) = old2New[mesh_dim][oldElem];
      else if (modified[oldElem].offset != -1) { //find new split element
        //Get ptcl position
        Vector<mesh_dim> pos;
        for (int i = 0; i<mesh_dim; i++)
          pos[i] = pPos(pid,i);

        //Get parent element
        auto key = modified[oldElem].key;
        auto oldChild = getChildElem(old_downward, pDim(pid), pParent(pid), pChild(pid));
        auto spltEdge = keys2edges[key];
        auto spltEdgeIdx = code_which_down(modified[oldElem].code);
        auto spltVertIdx = simplex_opposite_template(mesh_dim, EDGE, spltEdgeIdx);
        auto oldVerts = gather_verts<mesh_dim+1>(old_downward[0].ab2b, LO(oldElem));
        auto oldCoords = gather_vectors<mesh_dim+1,mesh_dim>(old_verts2coords, oldVerts);
        auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(pos, oldCoords); //TODO: account for flipping in 3D cases

        int rotation = code_rotation(modified[oldElem].code);
        auto highIdx = simplex_down_template(mesh_dim, EDGE, spltEdgeIdx, 0 ^ rotation);
        auto lowIdx = simplex_down_template(mesh_dim, EDGE, spltEdgeIdx, 1 ^ rotation);
        auto target = baryCoords[lowIdx] > baryCoords[highIdx] ? 0 : 1;
        bool onSplit = are_close(baryCoords[highIdx], baryCoords[lowIdx]);
        if (onSplit) target = 0;
        auto prod = keys2prods[key] + modified[oldElem].offset*2 + target;

        auto keptSide = simplex_opposite_template(mesh_dim, VERT, (target == 0) ? highIdx : lowIdx);
        Int old2NewIdx[mesh_dim+1] = {0};
        for (Int newIdx = 0; newIdx < mesh_dim; ++newIdx) {
          auto oldIdx = simplex_down_template(mesh_dim, mesh_dim - 1, keptSide, newIdx);
          old2NewIdx[oldIdx] = newIdx;
        }
        pParent(pid) = prods2new_ents[prod];

        if (onSplit && are_close(baryCoords[spltVertIdx], 0)){ //case 1: ptcl on the new vert
          pChild(pid) = mesh_dim;
          pDim(pid) = 0;
          return; //go to next ptcl
        }
        if (pDim(pid) == 0 && are_close(baryCoords[pChild(pid)], 1)) { //case 2: ptcl stayed on a vert
          pChild(pid) = old2NewIdx[pChild(pid)];
          auto newVert = getChildElem(pid);
          auto firstElemIdx = new_upward[0].a2ab[newVert];
          setPtcl(pid, 0, new_upward[0].ab2b[firstElemIdx], newVert);
          return; //go to next ptcl
        }
        if (onSplit) { //case 3: ptcl on new edge
          auto edgeIdx = pParent(pid)*simplex_degree(mesh_dim, 1) + rotation + spltEdgeIdx;
          pChild(pid) = edgeIdx;
          pDim(pid) = 1;
          return; //go to next ptcl
        }
        for (int i=0; i<simplex_degree(mesh_dim, 1); i++) //case 4: ptcl on old edge
          if (are_close(baryCoords[i], 0)) {
            auto newEdge = old2New[1][oldChild];
            auto firstElemIdx = new_upward[1].a2ab[newEdge];
            pParent(pid) = new_upward[1].ab2b[firstElemIdx];
            pChild(pid) = getChildIndex(1, pParent(pid), newEdge);
            pDim(pid) = 1;
            return;
          }
      }
      else {
        printf("WARNING: element skipped during particle adaptation\n");
      }
    });
  }
  virtual void coarsen(Mesh& old_mesh, Mesh& new_mesh, LOs keys2verts,
      Adj keys2doms, Int prod_dim, LOs prods2new_ents, 
      LOs same_ents2old_ents, LOs same_ents2new_ents) {
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