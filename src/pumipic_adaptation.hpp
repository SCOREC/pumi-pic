#ifndef PUMIPIC_ADAPTATION_HPP
#define PUMIPIC_ADAPTATION_HPP

#include "pumipic_kktypes.hpp"
#include "Omega_h_align.hpp"
#include "Omega_h_scalar.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_shape.hpp"
#include <MemberTypeLibraries.h>

using particle_structs::MemberTypes;
enum MemberIndex{POS, PARENT, CHILD, DIM, PID};
typedef MemberTypes<double[3], Omega_h::LO, Omega_h::Int, Omega_h::Int, int> Type;
typedef Kokkos::DefaultExecutionSpace ExeSpace;
typedef ps::ParticleStructure<Type,ExeSpace> PS;

namespace Omega_h {
  template<int mesh_dim>
  struct ParticleAdapt : public UserTransfer {

  struct ModifiedElem {
    LO key=-1;
    LO offset=-1;
    LO code=-1;

    ModifiedElem() : key(-1), offset(-1), code(-1) {}
    ModifiedElem(LO k, LO o, LO c) : key(k), offset(o), code(c) {}
  };

  PS*& ptcls;
  Mesh& mesh;
  Adj upward[mesh_dim];
  Adj downward[mesh_dim];
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
      upward[i] = meshIn.ask_up(i, mesh_dim);
      downward[i] = meshIn.ask_down(mesh_dim, i);
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
        if (downward[dim].ab2b[parent*degree + i] == child) childIdx = i;

    pDim(pid) = dim;
    pParent(pid) = parent;
    pChild(pid) = childIdx;
  }

  KOKKOS_INLINE_FUNCTION
  LO getLowestParent(LO child, Int dim) const {
    if (dim == mesh_dim) return child;
    auto lowestParentIdx = upward[dim].a2ab[child];
    return upward[dim].ab2b[lowestParentIdx];
  }

  KOKKOS_INLINE_FUNCTION
  LO getChildElem(LO pid) const {
    if (pDim(pid) == mesh_dim) return pParent(pid);
    auto degree = simplex_degree(mesh_dim, pDim(pid));
    return downward[pDim(pid)].ab2b[pParent(pid)*degree + pChild(pid)];
  }

  KOKKOS_INLINE_FUNCTION
  Int flip(Int index) const {
    if (mesh_dim < 3) return index;
    if (index == 1) return 2;
    if (index == 2) return 1;
    return index;
  }

  KOKKOS_INLINE_FUNCTION
  Int faceVertOppositeEdge(const Int face, const Kokkos::Array<Int, 2> edgeVerts) const {
    for (int i = 0; i < 3; i++){
      auto v = (mesh_dim == 2) ? i : simplex_down_template(mesh_dim, FACE, face, i);
      if (v != edgeVerts[0] && v != edgeVerts[1]) return v;
    }
    return -1;
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

  template <Int dim>
  constexpr Kokkos::Array<Int, dim+1> get_indices(Int index, Int rotation = 0) const {
    Kokkos::Array<Int, dim+1> output;
    for (int i=0; i<dim+1; i++) output[i] = simplex_down_template(mesh_dim, dim, index, i ^ rotation);
    return output;
  }

  Write<LO> getUnchanged(Mesh& old_mesh, Int dim, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    Write<LO> old2New(old_mesh.nents(dim), -1);
    parallel_for(same_ents2old_ents.size(), OMEGA_H_LAMBDA(LO i) {
      LO oldElem = same_ents2old_ents[i];
      old2New[oldElem] = same_ents2new_ents[i];
    });
    return old2New;
  }

  void update2LowestParent(const LO pid) const {
    if (pDim(pid) == mesh_dim) return;
    auto newChild = getChildElem(pid);
    auto lowestParent = getLowestParent(newChild, pDim(pid));
    setPtcl(pid, pDim(pid), lowestParent, newChild);
  }

  //TODO: Temporary remove
  template <Int n>
  void how_barycentric_inside(Vector<n> xi, Real& min, Real& max) const {
    min = reduce(xi, minimum<Real>());
    max = reduce(xi, maximum<Real>());
  }

  Kokkos::View<ModifiedElem*> gatherModified(LOs keys2entity, Int dim) {
    auto entity2elem = mesh.ask_up(dim, mesh_dim);
    Kokkos::View<ModifiedElem*> modified("modified_elems", mesh.nelems());
    parallel_for(keys2entity.size(), KOKKOS_CLASS_LAMBDA(LO key) {
      LO ent = keys2entity[key];
      auto elem_begin = entity2elem.a2ab[ent];
      auto elem_end = entity2elem.a2ab[ent + 1];
      for (auto idx = elem_begin; idx < elem_end; ++idx) {
        auto elem = entity2elem.ab2b[idx];
        modified[elem] = ModifiedElem(key, idx-elem_begin, entity2elem.codes[idx]);
      }
    });
    return modified;
  }

  virtual void refine(Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges,
      LOs keys2midverts, Int prod_dim, LOs keys2prods, LOs prods2new_ents,
      LOs same_ents2old_ents, LOs same_ents2new_ents) {

    if (prod_dim != mesh_dim) return;
    auto old2New = getUnchanged(old_mesh, prod_dim, same_ents2old_ents, same_ents2new_ents);
    auto modified = gatherModified(keys2edges, EDGE);
    auto old_cell2verts = old_mesh.ask_down(mesh_dim, 0).ab2b;
    auto old_vert2coords = old_mesh.coords();
    update(new_mesh);

    //Update modified elements
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const LO pid) {
      auto oldElem = pParent(pid);
      if (old2New[oldElem] != -1) //update unchanged element id
        pParent(pid) = old2New[oldElem];
      else if (modified[oldElem].offset != -1) { //find new split element
        auto newVert = mesh_dim;
        auto rotation = code_rotation(modified[oldElem].code);
        auto spltEdgeIdx = code_which_down(modified[oldElem].code);
        auto spltVerts = get_indices<EDGE>(spltEdgeIdx, rotation);
        auto oldVerts = gather_verts<mesh_dim+1>(old_cell2verts, LO(oldElem));
        auto oldCoords = gather_vectors<mesh_dim+1,mesh_dim>(old_vert2coords, oldVerts);
        auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), oldCoords);
        bool onSplit = are_close(baryCoords[spltVerts[0]], baryCoords[spltVerts[1]]) && !are_close(baryCoords[spltVerts[0]], 0);
        auto target = (onSplit || baryCoords[spltVerts[1]] > baryCoords[spltVerts[0]]) ? 0 : 1;
        auto prod = keys2prods[modified[oldElem].key] + modified[oldElem].offset*2 + target;
        pParent(pid) = prods2new_ents[prod];

        auto keptSide = simplex_opposite_template(mesh_dim, VERT, spltVerts[target]);
        Int old2NewIdx[mesh_dim+1] = {0}; //one elem kept blank
        for (Int newIdx = 0; newIdx < mesh_dim; ++newIdx) {
          auto oldIdx = simplex_down_template(mesh_dim, mesh_dim - 1, keptSide, newIdx);
          old2NewIdx[oldIdx] = flip(newIdx);
        }

        if (onSplit) pDim(pid) = pDim(pid) - 1;
        if (onSplit && pDim(pid) == 0) { //ptcl on the new vert
          pChild(pid) = newVert;
        }
        else if (onSplit && pDim(pid) == 1) { //ptcl on a new edge
          auto oppositeVert = faceVertOppositeEdge(pChild(pid), spltVerts);
          pChild(pid) = verts2Edge(newVert, old2NewIdx[oppositeVert]);
        }
        else if (onSplit && pDim(pid) == 2) { //particle on a new face
          auto oppositeEdge = simplex_opposite_template(mesh_dim, EDGE, spltEdgeIdx);
          auto oppositeVerts = get_indices<EDGE>(oppositeEdge);
          auto edge1 = verts2Edge(newVert, old2NewIdx[oppositeVerts[0]]);
          auto edge2 = verts2Edge(newVert, old2NewIdx[oppositeVerts[1]]);
          pChild(pid) = edges2Face(edge1, edge2);
        }
        else if (pDim(pid) == 0) { //ptcl stayed on same vert
          pChild(pid) = old2NewIdx[pChild(pid)];
        }
        else if (pDim(pid) == 1) { //ptcl stayed on same edge
          auto edgeVerts = get_indices<EDGE>(pChild(pid));
          pChild(pid) = (pChild(pid) == spltEdgeIdx) ? 
            verts2Edge(newVert, old2NewIdx[spltVerts[1-target]]) : //old edge was split
            verts2Edge(old2NewIdx[edgeVerts[0]], old2NewIdx[edgeVerts[1]]); //old edge stayed the same
        }
        else if (pDim(pid) == 2 && pDim(pid) < mesh_dim) { //particle stayed on face
          if (are_close(baryCoords[spltVerts[0]], 0) || are_close(baryCoords[spltVerts[1]], 0)) { //old face stayed the same
            auto faceVerts = get_indices<FACE>(pChild(pid));
            auto edge1 = verts2Edge(old2NewIdx[faceVerts[0]], old2NewIdx[faceVerts[1]]);
            auto edge2 = verts2Edge(old2NewIdx[faceVerts[0]], old2NewIdx[faceVerts[2]]);
            pChild(pid) = edges2Face(edge1, edge2);
          }
          else { //old face was split
            auto oppositeVert = faceVertOppositeEdge(pChild(pid), spltVerts);
            auto edge1 = verts2Edge(newVert, old2NewIdx[spltVerts[1-target]]);
            auto edge2 = verts2Edge(newVert, old2NewIdx[oppositeVert]);
            pChild(pid) = edges2Face(edge1, edge2);
          }
        }
        update2LowestParent(pid);
      }
      else printf("WARNING: element skipped during particle adaptation\n");
    });
  }

  void updatePtclsCavitySearch(Mesh& old_mesh, Mesh& new_mesh, LOs keys2prods, LOs prods2new_ents,  
      LOs same_ents2old_ents, LOs same_ents2new_ents, Kokkos::View<ModifiedElem*> modified_elem) {

    auto old2New = getUnchanged(old_mesh, mesh_dim, same_ents2old_ents, same_ents2new_ents);
    auto vert2coords = new_mesh.coords();
    update(new_mesh);

    //Update modified elements
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto oldElem = pParent(pid);
      if (old2New[oldElem] != -1) { //update unchanged element id
        pParent(pid) = old2New[oldElem];
        update2LowestParent(pid);
      }
      else if (modified_elem[oldElem].key != -1) {
        auto key = modified_elem[oldElem].key;
        auto elem_begin = keys2prods[key];
        auto elem_end = keys2prods[key+1];
        for (auto idx = elem_begin; idx < elem_end; ++idx) {
          auto newElem = prods2new_ents[idx];
          auto verts = gather_verts<mesh_dim+1>(downward[0].ab2b, LO(newElem));
          auto coords = gather_vectors<mesh_dim+1,mesh_dim>(vert2coords, verts);
          auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), coords);
          if (!is_barycentric_inside(baryCoords, EPSILON)) continue;
          pParent(pid) = newElem;
          pDim(pid) = mesh_dim;

          for (Int dim = 0; dim < mesh_dim; dim++)
          for (Int ent = 0; ent < simplex_degree(mesh_dim, dim); ent++) {
            Real baryCoordsSum = 0.0;
            for (Int vert = 0; vert < simplex_degree(dim, VERT); vert++) {
              auto vertIdx = simplex_down_template(mesh_dim, dim, ent, vert);
              if (are_close(baryCoords[vertIdx], 0)) {baryCoordsSum = -100.0; break;}
              else baryCoordsSum += baryCoords[vertIdx];
            }
            if (!are_close(baryCoordsSum, 1.0)) continue;
            pChild(pid) = ent;
            pDim(pid) = dim;
            update2LowestParent(pid);
            return;
          }
          return;
        }
        printf("WARNING: no coarsen element found for particle %d\n", pid); //TODO Customize
      }
      else printf("WARNING: particle %d skipped during particle adaptation coarsening\n", pid);
    });
  }

  virtual void coarsen(Mesh& old_mesh, Mesh& new_mesh, LOs keys2verts,
      Adj keys2doms, Int prod_dim, LOs prods2new_ents, 
      LOs same_ents2old_ents, LOs same_ents2new_ents) {

    if (prod_dim != mesh_dim) return;
    auto modified_elem = gatherModified(keys2verts, VERT);
    updatePtclsCavitySearch(old_mesh, new_mesh, keys2doms.a2ab, prods2new_ents, same_ents2old_ents, same_ents2new_ents, modified_elem);
  }

  virtual void swap(Mesh& old_mesh, Mesh& new_mesh, Int prod_dim,
      LOs keys2edges, LOs keys2prods, LOs prods2new_ents,
      LOs same_ents2old_ents, LOs same_ents2new_ents) {

    if (prod_dim != mesh_dim) return;
    auto modified_elem = gatherModified(keys2edges, EDGE);
    updatePtclsCavitySearch(old_mesh, new_mesh, keys2prods, prods2new_ents, same_ents2old_ents, same_ents2new_ents, modified_elem);
  }

  virtual void swap_copy_verts(Mesh& old_mesh, Mesh& new_mesh) {};
};

}//namespace
#endif //define