#ifndef PUMIPIC_ADAPTATION_HPP
#define PUMIPIC_ADAPTATION_HPP

#include "pumipic_kktypes.hpp"
#include "Omega_h_align.hpp"
#include "Omega_h_scalar.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_adapt.hpp"
#include "pumipic_utils.hpp"
#include <MemberTypeLibraries.h>

namespace pp = pumipic;

#if defined(OMEGA_H_USE_EGADS) || defined (OMEGA_H_USE_EGADSLITE)
#define PP_ENABLE_SNAP
#endif

namespace Omega_h {

namespace {
  constexpr OMEGA_H_INLINE Int flip_new_vert(const Int dim, const Int index) {
    if (dim < 3) return index;
    if (index == 1) return 2;
    if (index == 2) return 1;
    return index;
  }

  struct ModifiedElem {
    LO key=-1;
    LO offset=-1;
    LO code=-1;
    KOKKOS_INLINE_FUNCTION
    ModifiedElem() : key(-1), offset(-1), code(-1) {}
    KOKKOS_INLINE_FUNCTION
    ModifiedElem(LO k, LO o, LO c) : key(k), offset(o), code(c) {}
  };

  template<int mesh_dim>
  struct MeshData {
    Reals vert2coords;
    Adj upward[mesh_dim];
    Adj downward[mesh_dim];
    Read<I8> class_dim[mesh_dim];
    Read<ClassId> class_id[mesh_dim];

    MeshData(Mesh& mesh) {
      vert2coords = mesh.coords();
      for (int i=0; i<mesh_dim; i++) {
        upward[i] = mesh.ask_up(i, mesh_dim);
        downward[i] = mesh.ask_down(mesh_dim, i);
        class_dim[i] = mesh.get_array<Omega_h::I8>(i, "class_dim");
        class_id[i] = mesh.get_array<Omega_h::ClassId>(i, "class_id");
      }
    }
  };
}

template<int mesh_dim, typename PS, int POS, int PARENT, int CHILD, int DIM>
struct ParticleAdapt : public UserTransfer {

  PS*& ptcls;
  Mesh& mesh;
  Reals vert2coords;
  Adj upward[mesh_dim];
  Adj downward[mesh_dim];
  Read<I8> class_dim[mesh_dim];
  Read<ClassId> class_id[mesh_dim];
  typename PS::template Slice<POS> pPos;
  typename PS::template Slice<PARENT> pParent;
  typename PS::template Slice<CHILD> pChild;
  typename PS::template Slice<DIM> pDim;

  ParticleAdapt(PS*& ptclsIn, Mesh& meshIn) : ptcls(ptclsIn), mesh(meshIn) {
    update(meshIn);
  }

  void update(Mesh& meshIn) {
    pPos = ptcls->template get<POS>();
    pParent = ptcls->template get<PARENT>();
    pChild = ptcls->template get<CHILD>();
    pDim = ptcls->template get<DIM>();
    vert2coords = meshIn.coords();
    for (int i=0; i<mesh_dim; i++) {
      upward[i] = meshIn.ask_up(i, mesh_dim);
      downward[i] = meshIn.ask_down(mesh_dim, i);
      class_dim[i] = meshIn.get_array<Omega_h::I8>(i, "class_dim");
      class_id[i] = meshIn.get_array<Omega_h::ClassId>(i, "class_id");
    }
  }

  OMEGA_H_DEVICE Vector<mesh_dim> getPos(const LO pid) const {
    Vector<mesh_dim> pos;
    for (int i = 0; i<mesh_dim; i++) pos[i] = pPos(pid,i);
    return pos;
  }

  OMEGA_H_DEVICE void setPtcl(const LO pid, const Int dim, const LO parent, const LO child) const {
    auto nEnts = simplex_degree(mesh_dim, dim);
    int childIdx = -1;
    if (dim != mesh_dim)
      for (auto i = 0; i < nEnts; i++)
        if (downward[dim].ab2b[parent*nEnts + i] == child) childIdx = i;

    pDim(pid) = dim;
    pParent(pid) = parent;
    pChild(pid) = childIdx;
  }

  OMEGA_H_DEVICE LO getLowestParent(const LO child, const Int dim) const {
    if (dim == mesh_dim) return child;
    auto lowestParentIdx = upward[dim].a2ab[child];
    return upward[dim].ab2b[lowestParentIdx];
  }

  OMEGA_H_DEVICE LO getChildElem(const LO pid, const Adj down[mesh_dim]) const {
    if (pDim(pid) == mesh_dim) return pParent(pid);
    auto nEnts = simplex_degree(mesh_dim, pDim(pid));
    return down[pDim(pid)].ab2b[pParent(pid)*nEnts + pChild(pid)];
  }

  OMEGA_H_DEVICE LO getChildElem(const LO pid) const {
  return getChildElem(pid, downward);
}

  OMEGA_H_DEVICE void update2LowestParent(const LO pid) const {
    if (pDim(pid) == mesh_dim) return;
    auto newChild = getChildElem(pid);
    auto lowestParent = getLowestParent(newChild, pDim(pid));
    setPtcl(pid, pDim(pid), lowestParent, newChild);
  }

  static Write<LO> getUnchanged(Mesh& old_mesh, const Int dim, const LOs same_ents2old_ents, const LOs same_ents2new_ents) {
    Write<LO> old2New(old_mesh.nents(dim), -1);
    parallel_for(same_ents2old_ents.size(), OMEGA_H_LAMBDA(LO i) {
      LO oldElem = same_ents2old_ents[i];
      old2New[oldElem] = same_ents2new_ents[i];
    });
    return old2New;
  }

  static Kokkos::View<ModifiedElem*> gatherModified(Mesh& mesh, const LOs keys2entity, const Int dim) {
    auto entity2elem = mesh.ask_up(dim, mesh_dim);
    Kokkos::View<ModifiedElem*> modified("modified_elems", mesh.nelems());
    parallel_for(keys2entity.size(), OMEGA_H_LAMBDA(LO key) {
      LO ent = keys2entity[key];
      auto elem_begin = entity2elem.a2ab[ent];
      for (auto idx = elem_begin; idx < entity2elem.a2ab[ent + 1]; ++idx) {
        auto elem = entity2elem.ab2b[idx];
        modified[elem] = ModifiedElem(key, idx-elem_begin, entity2elem.codes[idx]);
      }
    });
    return modified;
  }

  OMEGA_H_DEVICE Real barycentric_distance(const LO pid, const LO elem) const {
    auto verts = gather_verts<mesh_dim+1>(downward[VERT].ab2b, elem);
    auto coords = gather_vectors<mesh_dim+1,mesh_dim>(vert2coords, verts);
    auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), coords);
    baryCoords = pp::clamp_barycentric<mesh_dim>(baryCoords);
    auto newPosition = pp::global_from_barycentric<mesh_dim,mesh_dim>(baryCoords, coords);
    return norm(newPosition - getPos(pid));
  }

  OMEGA_H_DEVICE void assign2Elem(const LO pid, const LO elem) const {
    auto verts = gather_verts<mesh_dim+1>(downward[VERT].ab2b, elem);
    auto coords = gather_vectors<mesh_dim+1,mesh_dim>(vert2coords, verts);
    auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), coords);
    OMEGA_H_CHECK(is_barycentric_inside(baryCoords, EPSILON));
    pParent(pid) = elem;
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
      pDim(pid) = dim;
      pChild(pid) = ent;
    }
    update2LowestParent(pid);
  }

  void populateFields() {
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const LO pid) {
      assign2Elem(pid, pParent(pid));
    });
  }

  OMEGA_H_DEVICE void snap2Surface(const I8 old_class_dim, const LO pid, const LO elem) const {
    #ifdef PP_ENABLE_SNAP
    auto verts = gather_verts<mesh_dim+1>(downward[VERT].ab2b, elem);
    auto coords = gather_vectors<mesh_dim+1,mesh_dim>(vert2coords, verts);
    auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), coords);
    if (old_class_dim == mesh_dim && is_barycentric_inside(baryCoords)) return;
    //TODO: Right now this is an approximation because we don't have access to Omega_h paramteric coordinates.
    //The ideal solution would be to snap the particle to the surface of the model using parametric
    //coordinates and then use barycentric coordinates to move the particle to the surface of the mesh.
    if (pDim(pid) < mesh_dim && is_barycentric_inside(baryCoords)) {
      Int closest = 0;
      for (Int i=1; i<mesh_dim+1; i++)
        if (baryCoords[i] < baryCoords[closest]) closest = i;
      baryCoords[closest] = 0;
    }
    baryCoords = pp::clamp_barycentric<mesh_dim>(baryCoords);
    auto newPosition = pp::global_from_barycentric<mesh_dim,mesh_dim>(baryCoords, coords);
    for (Int i=0; i<mesh_dim; i++) pPos(pid, i) = newPosition[i];
    #endif
  }

  virtual void refine(Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges, LOs keys2midverts, Int prod_dim, 
      LOs keys2prods, LOs prods2new_ents, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    if (prod_dim != mesh_dim) return;
    auto old2New = getUnchanged(old_mesh, prod_dim, same_ents2old_ents, same_ents2new_ents);
    auto modified = gatherModified(old_mesh, keys2edges, EDGE);
    auto old_cell2verts = old_mesh.ask_down(mesh_dim, VERT).ab2b;
    auto old_vert2coords = old_mesh.coords();
    update(new_mesh);

    //Update modified elements
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const LO pid) {
      auto oldElem = pParent(pid);
      if (old2New[oldElem] != -1) {//update unchanged element id
        pParent(pid) = old2New[oldElem];
        update2LowestParent(pid);
      }
      else if (modified[oldElem].offset != -1) { //find new split element
        auto newVert = mesh_dim;
        auto rotation = code_rotation(modified[oldElem].code);
        auto spltEdgeIdx = code_which_down(modified[oldElem].code);
        auto spltVerts = ps::simplex_gather_down<EDGE>(mesh_dim, spltEdgeIdx, rotation);
        auto oldVerts = gather_verts<mesh_dim+1>(old_cell2verts, oldElem);
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
          old2NewIdx[oldIdx] = flip_new_vert(mesh_dim, newIdx);
        }

        if (onSplit) pDim(pid) = pDim(pid) - 1;
        if (onSplit && pDim(pid) == VERT) { //ptcl on the new vert
          pChild(pid) = newVert;
        }
        else if (onSplit && pDim(pid) == EDGE) { //ptcl on a new edge
          auto oppositeVert = pp::face_vertex_opposite_edge(mesh_dim, pChild(pid), spltEdgeIdx);
          pChild(pid) = pp::edge_from_verts(mesh_dim, newVert, old2NewIdx[oppositeVert]);
        }
        else if (onSplit && pDim(pid) == FACE) { //particle on a new face
          auto oppositeEdge = simplex_opposite_template(mesh_dim, EDGE, spltEdgeIdx);
          auto oppositeVerts = pp::simplex_gather_down<EDGE>(mesh_dim, oppositeEdge);
          auto edge1 = pp::edge_from_verts(mesh_dim, newVert, old2NewIdx[oppositeVerts[0]]);
          auto edge2 = pp::edge_from_verts(mesh_dim, newVert, old2NewIdx[oppositeVerts[1]]);
          pChild(pid) = pp::face_from_edges(edge1, edge2);
        }
        else if (pDim(pid) == VERT) { //ptcl stayed on same vert
          pChild(pid) = old2NewIdx[pChild(pid)];
        }
        else if (pDim(pid) == EDGE) { //ptcl stayed on same edge
          auto edgeVerts = pp::simplex_gather_down<EDGE>(mesh_dim, pChild(pid));
          pChild(pid) = (pChild(pid) == spltEdgeIdx) ? 
            pp::edge_from_verts(mesh_dim, newVert, old2NewIdx[spltVerts[1-target]]) : //old edge was split
            pp::edge_from_verts(mesh_dim, old2NewIdx[edgeVerts[0]], old2NewIdx[edgeVerts[1]]); //old edge stayed the same
        }
        else if (pDim(pid) == FACE && pDim(pid) < mesh_dim) { //particle stayed on face
          if (are_close(baryCoords[spltVerts[0]], 0) || are_close(baryCoords[spltVerts[1]], 0)) { //old face stayed the same
            auto faceVerts = pp::simplex_gather_down<FACE>(mesh_dim, pChild(pid));
            auto edge1 = pp::edge_from_verts(mesh_dim, old2NewIdx[faceVerts[0]], old2NewIdx[faceVerts[1]]);
            auto edge2 = pp::edge_from_verts(mesh_dim, old2NewIdx[faceVerts[0]], old2NewIdx[faceVerts[2]]);
            pChild(pid) = pp::face_from_edges(edge1, edge2);
          }
          else { //old face was split
            auto oppositeVert = pp::face_vertex_opposite_edge(mesh_dim, pChild(pid), spltEdgeIdx);
            auto edge1 = pp::edge_from_verts(mesh_dim, newVert, old2NewIdx[spltVerts[1-target]]);
            auto edge2 = pp::edge_from_verts(mesh_dim, newVert, old2NewIdx[oppositeVert]);
            pChild(pid) = pp::face_from_edges(edge1, edge2);
          }
        }
        update2LowestParent(pid);
      }
      else Kokkos::abort("[ERROR] : element skipped during particle adaptation\n");
    });
  }

  void updatePtclsCavitySearch(Mesh& old_mesh, Mesh& new_mesh, LOs keys2prods, LOs prods2new_ents,  
      LOs same_ents2old_ents, LOs same_ents2new_ents, Kokkos::View<ModifiedElem*> modified_elem) {
    update(new_mesh);
    auto old2New = getUnchanged(old_mesh, mesh_dim, same_ents2old_ents, same_ents2new_ents);
    MeshData old_data = MeshData<mesh_dim>(old_mesh);
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto oldElem = pParent(pid);
      auto newElem = oldElem;
      if (old2New[oldElem] != -1)
        newElem = old2New[oldElem];
      else if (modified_elem[oldElem].key != -1) {
        Real closest = 9999999;
        auto key = modified_elem[oldElem].key;
        for (auto idx = keys2prods[key]; idx < keys2prods[key+1]; ++idx) {
          auto dist = barycentric_distance(pid, prods2new_ents[idx]);
          if (dist < closest) {closest = dist; newElem = prods2new_ents[idx];}
        }
      }
      else Kokkos::abort("[ERROR] : particle skipped during particle adaptation of swap/coarsen\n");

      auto oldChild = getChildElem(pid, old_data.downward);
      auto oldClassDim = old_data.class_dim[pDim(pid)][oldChild];
      snap2Surface(oldClassDim, pid, newElem);
      assign2Elem(pid, newElem);
    });
  }

  virtual void snap(Mesh& mesh, const Omega_h::Reals& old_vert2coords, const Omega_h::Reals& warp) {
    update(mesh);
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto elem = pParent(pid);
      auto child = getChildElem(pid);
      auto verts = gather_verts<mesh_dim+1>(downward[VERT].ab2b, elem);
      auto oldCoords = gather_vectors<mesh_dim+1,mesh_dim>(old_vert2coords, verts);
      auto newCoords = gather_vectors<mesh_dim+1,mesh_dim>(vert2coords, verts);
      auto oldBaryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), oldCoords);
      auto newBaryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), newCoords);
      bool insideAfterSnap = is_barycentric_inside(newBaryCoords, EPSILON);
      if (!insideAfterSnap || class_dim[pDim(pid)][child] < mesh_dim) {
        auto newPosition = pp::global_from_barycentric<mesh_dim,mesh_dim>(oldBaryCoords, newCoords);
        for (int i=0; i<mesh_dim; i++) pPos(pid, i) = newPosition[i];
      }
    });
  }

  virtual void coarsen(Mesh& old_mesh, Mesh& new_mesh, LOs keys2verts, Adj keys2doms, 
      Int prod_dim, LOs prods2new_ents, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    if (prod_dim != mesh_dim) return;
    auto modified_elem = gatherModified(old_mesh, keys2verts, VERT);
    updatePtclsCavitySearch(old_mesh, new_mesh, keys2doms.a2ab, prods2new_ents, same_ents2old_ents, same_ents2new_ents, modified_elem);
  }

  virtual void swap(Mesh& old_mesh, Mesh& new_mesh, Int prod_dim, LOs keys2edges, 
      LOs keys2prods, LOs prods2new_ents, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    if (prod_dim != mesh_dim) return;
    auto modified_elem = gatherModified(old_mesh, keys2edges, EDGE);
    updatePtclsCavitySearch(old_mesh, new_mesh, keys2prods, prods2new_ents, same_ents2old_ents, same_ents2new_ents, modified_elem);
  }

  virtual void swap_copy_verts(Mesh& old_mesh, Mesh& new_mesh) {};
};

}//namespace
#endif //define