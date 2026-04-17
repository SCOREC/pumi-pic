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
    LO rotation=-1;
  };

  PS* ptcls;
  int numEnt[mesh_dim];
  LOs old2New[mesh_dim+1];
  Adj new_upward[mesh_dim];
  Adj new_downward[mesh_dim];


  ParticleAdapt(PS* particles, Mesh& mesh) {
    ptcls = particles;
    for (int i=0; i<mesh_dim; i++)
      numEnt[i] = element_degree(mesh.family(), mesh_dim, i);
  }
 
  KOKKOS_INLINE_FUNCTION
  int getChildIndex(const Adj downward[mesh_dim], int dim, LO parent, LO child) const {
    for (auto i = 0; i < 3; i++)
      if (downward[dim].ab2b[parent*numEnt[dim] + i] == child) return i;
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  LO getChildElem(const Adj downward[mesh_dim], int dim, LO parent, int index) const {
    return downward[dim].ab2b[parent*numEnt[dim] + index];
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
        modified[elem].rotation = code_rotation(old_edge2Elem.codes[idx]);
      }
    });

    auto ptclPos = ptcls->get<POS>();
    auto ptclElem = ptcls->get<PARENT>();
    auto ptclChild = ptcls->get<CHILD>();
    auto ptclDim = ptcls->get<DIM>();
    auto old_verts2coords = old_mesh.coords();
    Adj old_downward[mesh_dim];
    for (int i=0; i<mesh_dim; i++) {
      new_upward[i] = new_mesh.ask_up(i, mesh_dim);
      new_downward[i] = new_mesh.ask_down(mesh_dim, i);
      old_downward[i] = old_mesh.ask_down(mesh_dim, i);
    }

    //Update modified elements
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto oldElem = ptclElem(pid);
      if (old2New[mesh_dim][oldElem] != -1) //update unchanged element id
        ptclElem(pid) = old2New[mesh_dim][oldElem];
      else if (modified[oldElem].offset != -1) { //find new split element
        auto key = modified[oldElem].key;
        auto oldChild = getChildElem(old_downward, ptclDim(pid), ptclElem(pid), ptclChild(pid));

        //Get ptcl position
        Vector<mesh_dim> pos;
        for (int i = 0; i<mesh_dim; i++)
          pos[i] = ptclPos(pid,i);

        //Get parent element
        // auto splitPos = get_vector<mesh_dim>(new_verts2coords, LO(keys2midverts[key]));
        auto oldVerts = gather_verts<mesh_dim+1>(old_downward[0].ab2b, LO(oldElem));
        auto oldCoords = gather_vectors<mesh_dim+1,mesh_dim>(old_verts2coords, oldVerts);
        auto baryCoords = barycentric_from_global<mesh_dim,mesh_dim>(pos, oldCoords);
        int side = baryCoords[0] > baryCoords[2] ? 0 : 2;
        if (are_close(baryCoords[0], baryCoords[2])) side = 1;
        const int rotateDirCode[3][2] = {{0,1},{0,0},{1,0}};
        int rotated = rotateDirCode[side][modified[oldElem].rotation];
        auto prod = keys2prods[key] + modified[oldElem].offset*2 + rotated;
        ptclElem(pid) = prods2new_ents[prod];
        ptclDim(pid) = mesh_dim;

        if (side == 1 && are_close(baryCoords[1], 0)){ //case 1: ptcl on new vert
          ptclChild(pid) = 1;
          ptclDim(pid) = 0;
          return; //go to next ptcl
        }
        for (int i=0; i<numEnt[0]; i++) //case 2: ptcl on old vert
          if (are_close(baryCoords[i], 1)) {
            auto newVert = old2New[0][oldVerts[i]];
            auto firstElemIdx = new_upward[0].a2ab[newVert];
            ptclElem(pid) = new_upward[0].ab2b[firstElemIdx];
            ptclChild(pid) = getChildIndex(new_downward, 0, ptclElem(pid), newVert);
            ptclDim(pid) = 0;
            return; //go to next ptcl
          }
        if (side == 1) { //case 3: ptcl on new edge
          // auto edgeIdx = ptclElem(pid)*nEdges + modified[oldElem].rotation + side;
          // ptclChild(pid) = edgeIdx;
          // ptclDim(pid) = 1;
          // return; //go to next ptcl
        }
        for (int i=0; i<numEnt[1]; i++) //case 4: ptcl on old edge
          if (are_close(baryCoords[i], 0)) {
            auto newEdge = old2New[1][oldChild];
            auto firstElemIdx = new_upward[1].a2ab[newEdge];
            ptclElem(pid) = new_upward[1].ab2b[firstElemIdx];
            ptclChild(pid) = getChildIndex(new_downward, 1, ptclElem(pid), newEdge);
            ptclDim(pid) = 1;
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