#ifndef PUMIPIC_ADAPTATION_HPP
#define PUMIPIC_ADAPTATION_HPP

#include "pumipic_kktypes.hpp"
#include "Omega_h_align.hpp"
#include "Omega_h_scalar.hpp"
#include "Omega_h_element.hpp"
#include <MemberTypeLibraries.h>

using particle_structs::MemberTypes;
enum MemberIndex{POS, PARENT, CHILD, DIM, PID};
typedef MemberTypes<double[3], Omega_h::LO, Omega_h::Int, Omega_h::Int, int> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef ps::ParticleStructure<Type,ExeSpace> PS;

namespace Omega_h {

  template <Int n>
  OMEGA_H_INLINE Real distance(Vector<n> a, Vector<n> b) OMEGA_H_NOEXCEPT {
    Real x = 0;
    for (Int i = 0; i < n; ++i) x += std::pow(a[i] - b[i], 2);
    return std::sqrt(x);
  }

  template<int mesh_dim>
  struct ParticleAdapt : public UserTransfer {

  struct ModifiedElem {
    LO key=-1;
    LO offset=-1;
    LO code=-1;
  };

  PS*& ptcls;
  Mesh& mesh;
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
  Vector<mesh_dim> getPos(LO pid) const {
    Vector<mesh_dim> pos;
    for (int i = 0; i<mesh_dim; i++) pos[i] = pPos(pid,i);
    return pos;
  }

  KOKKOS_INLINE_FUNCTION
  void setPtcl(LO pid, Int dim, LO parent, LO child) const {
    auto degree = simplex_degree(mesh_dim, dim);
    int childIdx = -1;
    if (dim != mesh_dim)
      for (auto i = 0; i < degree; i++)
        if (new_downward[dim].ab2b[parent*degree + i] == child) childIdx = i;

    pDim(pid) = dim;
    pParent(pid) = parent;
    pChild(pid) = childIdx;
  }

  KOKKOS_INLINE_FUNCTION
  LO getLowestParent(LO child, Int dim) const {
    if (dim == mesh_dim) return child;
    auto lowestParentIdx = new_upward[dim].a2ab[child];
    return new_upward[dim].ab2b[lowestParentIdx];
  }

  KOKKOS_INLINE_FUNCTION
  LO getChildElem(LO pid) const {
    if (pDim(pid) == mesh_dim) return pParent(pid);
    auto degree = simplex_degree(mesh_dim, pDim(pid));
    return new_downward[pDim(pid)].ab2b[pParent(pid)*degree + pChild(pid)];
  }

  KOKKOS_INLINE_FUNCTION
  Int flip(Int index) const {
    if (mesh_dim < 3) return index;
    if (index == 1) return 2;
    if (index == 2) return 1;
    return index;
  }

  KOKKOS_INLINE_FUNCTION
  Int edges2Face(Int edge1, Int edge2) const {
    for (int i=0; i<2; ++i) {
      auto face1 = simplex_up_template(mesh_dim, EDGE, edge1, i);
      for (int j=0; j<2; ++j) {
        auto face2 = simplex_up_template(mesh_dim, EDGE, edge2, j);
        if (face1.up == face2.up) return face1.up;
      }
    }
    return -1;
  }

  KOKKOS_INLINE_FUNCTION
  Int verts2Edge(Int vert1, Int vert2) const {
    for (int i=0; i<3; ++i) {
      auto edge1 = simplex_up_template(mesh_dim, VERT, vert1, i);
      for (int j=0; j<3; ++j) {
        auto edge2 = simplex_up_template(mesh_dim, VERT, vert2, j);
        if (edge1.up == edge2.up) return edge1.up;
      }
    }
    return -1;
  }

  Write<LO> getUnchanged(Mesh& old_mesh, Int dim, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    Write<LO> old2New(old_mesh.nents(dim), -1);
    parallel_for(same_ents2old_ents.size(), OMEGA_H_LAMBDA(LO i) {
      LO oldElem = same_ents2old_ents[i];
      old2New[oldElem] = same_ents2new_ents[i];
    });
    return old2New;
  }

  virtual void refine(Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges,
      LOs keys2midverts, Int prod_dim, LOs keys2prods, LOs prods2new_ents,
      LOs same_ents2old_ents, LOs same_ents2new_ents) {
    if (prod_dim != mesh_dim) return;
    auto old2New = getUnchanged(old_mesh, prod_dim, same_ents2old_ents, same_ents2new_ents);

    Kokkos::View<ModifiedElem*> modified("modified_elems", old_mesh.nelems());
    auto old_edge2Elem = old_mesh.ask_up(EDGE, mesh_dim);

    //Gather modified elements
    parallel_for(keys2edges.size(), KOKKOS_CLASS_LAMBDA(LO key) {
      LO edge = keys2edges[key];
      auto elem_begin = old_edge2Elem.a2ab[edge];
      auto elem_end = old_edge2Elem.a2ab[edge + 1];
      for (auto idx = elem_begin; idx < elem_end; ++idx) {
        auto elem = old_edge2Elem.ab2b[idx];
        modified[elem] = ModifiedElem(key, idx-elem_begin, old_edge2Elem.codes[idx]);
      }
    });

    update(new_mesh);
    auto old_vert2coords = old_mesh.coords();
    auto old_cell2verts = old_mesh.ask_down(mesh_dim, 0).ab2b;

    //Update modified elements
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto oldElem = pParent(pid);
      if (old2New[oldElem] != -1) //update unchanged element id
        pParent(pid) = old2New[oldElem];
      else if (modified[oldElem].offset != -1) { //find new split element
        auto rotation = code_rotation(modified[oldElem].code);
        auto spltEdgeIdx = code_which_down(modified[oldElem].code);
        Int edgeVerts[2]; Int spltVerts[2]; Int nonZeroVert = -1;
        for (int i=0; i<2; i++) edgeVerts[i] = simplex_down_template(mesh_dim, EDGE, pChild(pid), i ^ rotation);
        for (int i=0; i<2; i++) spltVerts[i] = simplex_down_template(mesh_dim, EDGE, spltEdgeIdx, i ^ rotation);
        auto oldVerts = gather_verts<mesh_dim+1>(old_cell2verts, LO(oldElem));
        auto oldCoords = gather_vectors<mesh_dim+1,mesh_dim>(old_vert2coords, oldVerts);
        auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), oldCoords);
        bool onSplit = are_close(baryCoords[spltVerts[0]], baryCoords[spltVerts[1]]) && !are_close(baryCoords[spltVerts[0]], 0);
        auto target = (onSplit || baryCoords[spltVerts[1]] > baryCoords[spltVerts[0]]) ? 0 : 1;
        for (Int i=0; i<mesh_dim+1; i++) 
          if (i != spltVerts[0] && i != spltVerts[1] && !are_close(baryCoords[i], 0))
            nonZeroVert = i;
        auto prod = keys2prods[modified[oldElem].key] + modified[oldElem].offset*2 + target;
        pParent(pid) = prods2new_ents[prod];

        auto keptSide = simplex_opposite_template(mesh_dim, VERT, spltVerts[target]);
        Int old2NewIdx[mesh_dim+1] = {0}; //one elem kept blank
        for (Int newIdx = 0; newIdx < mesh_dim; ++newIdx) {
          auto oldIdx = simplex_down_template(mesh_dim, mesh_dim - 1, keptSide, newIdx);
          old2NewIdx[oldIdx] = flip(newIdx);
        }

        if (pDim(pid) == 0 && are_close(baryCoords[pChild(pid)], 1)) { //ptcl stayed on same vert
          pChild(pid) = old2NewIdx[pChild(pid)];
        }
        else if (onSplit && are_close(baryCoords[spltVerts[1]]+baryCoords[spltVerts[0]], 1)){ //ptcl on the new vert
          pChild(pid) = mesh_dim;
          pDim(pid) = 0;
        }
        else if (onSplit && pDim(pid) == 2) { //ptcl on a new edge
          pChild(pid) = verts2Edge(mesh_dim, old2NewIdx[nonZeroVert]);
          pDim(pid) = 1;
        }
        else if (pDim(pid) == 1 && are_close(baryCoords[edgeVerts[0]]+baryCoords[edgeVerts[1]], 1)) { //ptcl stayed on same edge
          pChild(pid) = (pChild(pid) == spltEdgeIdx) ? 
            verts2Edge(mesh_dim, old2NewIdx[spltVerts[1-target]]) : 
            verts2Edge(old2NewIdx[edgeVerts[0]], old2NewIdx[edgeVerts[1]]);
        }
        else if (pDim(pid) == 2 && pDim(pid) < mesh_dim) { //particle stayed on face
          Int faceVerts[3]; Int edge1; Int edge2;
          for (int i=0; i<3; i++) faceVerts[i] = simplex_down_template(mesh_dim, FACE, pChild(pid), i);
          if (are_close(baryCoords[spltVerts[0]], 0) || are_close(baryCoords[spltVerts[1]], 0)) {
            edge1 = verts2Edge(old2NewIdx[faceVerts[0]], old2NewIdx[faceVerts[1]]);
            edge2 = verts2Edge(old2NewIdx[faceVerts[0]], old2NewIdx[faceVerts[2]]);
          }
          else {
            edge1 = verts2Edge(mesh_dim, old2NewIdx[spltVerts[1-target]]);
            edge2 = verts2Edge(mesh_dim, old2NewIdx[nonZeroVert]);
          }
          pChild(pid) = edges2Face(edge1, edge2);
        }
        else if (onSplit && pDim(pid) == 3) { //particle on a new face
          pChild(pid) = 3;
          pDim(pid) = 2;
        }
        if (pDim(pid) < mesh_dim) { //update parent to lowest adjacent
          auto newChild = getChildElem(pid);
          auto lowestParent = getLowestParent(newChild, pDim(pid));
          setPtcl(pid, pDim(pid), lowestParent, newChild);
        }
      }
      else printf("WARNING: element skipped during particle adaptation\n");
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