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
  constexpr OMEGA_H_INLINE Int flip_new_vert(Int dim, Int index) {
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
}

template<int mesh_dim, typename PS, int POS, int PARENT, int CHILD, int DIM>
struct ParticleAdapt : public UserTransfer {

  PS*& ptcls;
  Mesh& mesh;
  AdaptOpts* opts;
  Reals vert2coords;
  Adj upward[mesh_dim];
  Adj downward[mesh_dim];
  typename PS::template Slice<POS> pPos;
  typename PS::template Slice<PARENT> pParent;
  typename PS::template Slice<CHILD> pChild;
  typename PS::template Slice<DIM> pDim;
  Read<I8> onSurface;
  bool should_snap;

  ParticleAdapt(PS*& ptclsIn, Mesh& meshIn, bool shouldSnap=false) : ptcls(ptclsIn), mesh(meshIn) {
    should_snap = shouldSnap;
    update(meshIn);
  }

  void update(Mesh& meshIn) {
    pPos = ptcls->template get<POS>();
    pParent = ptcls->template get<PARENT>();
    pChild = ptcls->template get<CHILD>();
    pDim = ptcls->template get<DIM>();
    vert2coords = meshIn.coords();
    onSurface = Omega_h::mark_exposed_sides(&meshIn);
    for (int i=0; i<mesh_dim; i++) {
      upward[i] = meshIn.ask_up(i, mesh_dim);
      downward[i] = meshIn.ask_down(mesh_dim, i);
    }
  }

  void setOpts(AdaptOpts* opts2) {
    opts = opts2;
  }

  //TODO: remove after PR
  void stopExecution() const {
    opts->should_refine = false;
    opts->should_coarsen = false;
    opts->should_swap = false;
    opts->should_coarsen_slivers = false;
    mesh.remove_tag(VERT, "target_metric");
    #ifdef PP_ENABLE_SNAP
    opts->egads_model = nullptr;
    #endif
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

  Write<LO> getUnchanged(Mesh& old_mesh, Int dim, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    Write<LO> old2New(old_mesh.nents(dim), -1);
    parallel_for(same_ents2old_ents.size(), OMEGA_H_LAMBDA(LO i) {
      LO oldElem = same_ents2old_ents[i];
      old2New[oldElem] = same_ents2new_ents[i];
    });
    return old2New;
  }

  KOKKOS_INLINE_FUNCTION
  void update2LowestParent(const LO pid) const {
    if (pDim(pid) == mesh_dim) return;
    auto newChild = getChildElem(pid);
    auto lowestParent = getLowestParent(newChild, pDim(pid));
    setPtcl(pid, pDim(pid), lowestParent, newChild);
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

  template <Int n>
  OMEGA_H_DEVICE Vector<n> clamp_barycentric(Vector<n> baryCoords) const {
    double barySum = 0;
    for (Int i=0; i<mesh_dim+1; i++) 
      (baryCoords[i] < 0) ? baryCoords[i] = 0 : barySum += baryCoords[i];
    for (Int i=0; i<mesh_dim+1; i++)
      baryCoords[i] = baryCoords[i] / barySum; //Make coords add up to one
    return baryCoords;
  }

  template <Int sdim, Int edim>
  OMEGA_H_INLINE Vector<sdim> global_from_barycentric(Vector<edim + 1> const& barycentric_coords,
      Few<Vector<sdim>, edim + 1> const& node_coords) const {
    const auto basis = simplex_basis<sdim, edim>(node_coords);
    Vector<edim> lambda;
    for (Int i = 0; i < edim; ++i) lambda[i] = barycentric_coords[i + 1];
    return node_coords[0] + basis * lambda;
  }

  KOKKOS_INLINE_FUNCTION
  Real assign2Elem(const LO pid, const LO elem) const {
    auto verts = gather_verts<mesh_dim+1>(downward[VERT].ab2b, LO(elem));
    auto coords = gather_vectors<mesh_dim+1,mesh_dim>(vert2coords, verts);
    auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), coords);
    baryCoords = clamp_barycentric(baryCoords);
    auto newPosition = global_from_barycentric<mesh_dim,mesh_dim>(baryCoords, coords);
    if (!are_close(newPosition, getPos(pid))) return norm(newPosition - getPos(pid));
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
      pChild(pid) = ent;
      pDim(pid) = dim;
      update2LowestParent(pid);
      return 0;
    }
    return 0;
  }

  OMEGA_H_INLINE
  void snap2Surface(const LO pid, const LO elem) const {
    #ifdef PP_ENABLE_SNAP
    pParent(pid) = elem;
    auto verts = gather_verts<mesh_dim+1>(downward[VERT].ab2b, LO(elem));
    auto coords = gather_vectors<mesh_dim+1,mesh_dim>(vert2coords, verts);
    auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), coords);
    auto prevCoords = baryCoords;
    baryCoords = clamp_barycentric(baryCoords);
    //TODO: only change position if it was pushed inside or outside the mesh
    auto newPosition = global_from_barycentric<mesh_dim,mesh_dim>(baryCoords, coords);
    for (Int i=0; i<mesh_dim; i++) pPos(pid, i) = newPosition[i];
    if (!are_close(assign2Elem(pid, elem), 0)) printf("[ERROR]: Snap failed before %f, %f, %f, %f after %f, %f, %f, %f\n", prevCoords[0], prevCoords[1], prevCoords[2], prevCoords[3], baryCoords[0], baryCoords[1], baryCoords[2], baryCoords[3]);
    #endif
  }

  void populateFields() {
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const LO pid) {
      auto elem = pParent(pid);
      if (!are_close(assign2Elem(pid, elem), 0)) printf("WARNING: PID %d not in elem %d\n", pid, elem);
    });
  }

  virtual void refine(Mesh& old_mesh, Mesh& new_mesh, LOs keys2edges, LOs keys2midverts, Int prod_dim, 
      LOs keys2prods, LOs prods2new_ents, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    if (prod_dim != mesh_dim) return;
    auto old2New = getUnchanged(old_mesh, prod_dim, same_ents2old_ents, same_ents2new_ents);
    auto modified = gatherModified(keys2edges, EDGE);
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
      else printf("WARNING: element skipped during particle adaptation\n");
    });
  }

  virtual void snap(Mesh& mesh, const Omega_h::Reals& old_vert2coords, const Omega_h::Reals& warp) {
    if (!should_snap) return;
    update(mesh);
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto lastElem = pParent(pid);
      auto verts = gather_verts<mesh_dim+1>(downward[VERT].ab2b, LO(lastElem));
      auto oldCoords = gather_vectors<mesh_dim+1,mesh_dim>(old_vert2coords, verts);
      auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(getPos(pid), oldCoords);
      auto newCoords = gather_vectors<mesh_dim+1,mesh_dim>(vert2coords, verts);
      auto newPosition = global_from_barycentric<mesh_dim,mesh_dim>(baryCoords, newCoords);
      //TODO: only change position if it was pushed inside or outside the mesh
      for (int i=0; i<mesh_dim; i++) pPos(pid, i) = newPosition[i];
    });
  }

  void updatePtclsCavitySearch(Mesh& old_mesh, Mesh& new_mesh, LOs keys2prods, LOs prods2new_ents,  
      LOs same_ents2old_ents, LOs same_ents2new_ents, Kokkos::View<ModifiedElem*> modified_elem, std::string name) {
    update(new_mesh);
    auto old2New = getUnchanged(old_mesh, mesh_dim, same_ents2old_ents, same_ents2new_ents);
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto oldElem = pParent(pid);
      if (old2New[oldElem] != -1) { //update unchanged element id
        pParent(pid) = old2New[oldElem];
        update2LowestParent(pid);
        snap2Surface(pid, pParent(pid));
      }
      else if (modified_elem[oldElem].key != -1) {
        auto key = modified_elem[oldElem].key;
        auto elem_begin = keys2prods[key];
        auto elem_end = keys2prods[key+1];
        Real closest = 1000000;
        LO closestIdx = 0;
        for (auto idx = elem_begin; idx < elem_end; ++idx) {
          auto newElem = prods2new_ents[idx];
          auto dist = assign2Elem(pid, newElem);
          if (are_close(dist, 0)) return;
          if (dist < closest) {closest = dist; closestIdx = idx;}
        }
        snap2Surface(pid, prods2new_ents[closestIdx]);
      }
      else printf("WARNING: particle %d skipped during particle adaptation %s\n", pid, name.c_str());
    });
  }

  virtual void coarsen(Mesh& old_mesh, Mesh& new_mesh, LOs keys2verts, Adj keys2doms, 
      Int prod_dim, LOs prods2new_ents, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    if (prod_dim != mesh_dim) return;
    auto modified_elem = gatherModified(keys2verts, VERT);
    updatePtclsCavitySearch(old_mesh, new_mesh, keys2doms.a2ab, prods2new_ents, same_ents2old_ents, same_ents2new_ents, modified_elem, "coarsen");
  }

  virtual void swap(Mesh& old_mesh, Mesh& new_mesh, Int prod_dim, LOs keys2edges, 
      LOs keys2prods, LOs prods2new_ents, LOs same_ents2old_ents, LOs same_ents2new_ents) {
    if (prod_dim != mesh_dim) return;
    auto modified_elem = gatherModified(keys2edges, EDGE);
    updatePtclsCavitySearch(old_mesh, new_mesh, keys2prods, prods2new_ents, same_ents2old_ents, same_ents2new_ents, modified_elem, "swap");
  }

  virtual void swap_copy_verts(Mesh& old_mesh, Mesh& new_mesh) {};
};

}//namespace
#endif //define