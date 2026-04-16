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
  LOs old_vert2new_vert;
  int numEnt[mesh_dim];

  ParticleAdapt(PS* particles, Mesh& mesh) {
    ptcls = particles;
    for (int i=0; i<mesh_dim; i++)
      numEnt[i] = element_degree(mesh.family(), mesh_dim, i);
  }
 
  KOKKOS_INLINE_FUNCTION
  int getChildIndex(Adj elem2Vert, LO parent, LO child) const {
    for (auto i = 0; i < 3; i++)
      if (elem2Vert.ab2b[parent*3 + i] == child) return i;
    return 0;
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
    
    if (prod_dim == 0) old_vert2new_vert = getUnchanged(old_mesh, prod_dim, same_ents2old_ents, same_ents2new_ents);
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

    auto ptclPos = ptcls->get<POS>();
    auto ptclElem = ptcls->get<PARENT>();
    auto ptclChild = ptcls->get<CHILD>();
    auto ptclDim = ptcls->get<DIM>();
    auto new_verts2coords = new_mesh.coords();
    auto old_verts2coords = old_mesh.coords();
    auto old_elem2verts = old_mesh.get_adj(mesh_dim, VERT).ab2b;
    auto new_elem2edge = new_mesh.get_adj(mesh_dim, EDGE);
    auto new_vert2elem = new_mesh.ask_up(VERT, mesh_dim);
    auto new_elem2vert = new_mesh.get_adj(mesh_dim, VERT);

    //Update modified elements
    Kokkos::parallel_for(ptcls->nPtcls(), KOKKOS_CLASS_LAMBDA(const int pid) {
      auto oldElem = ptclElem(pid);
      if (same_old2New[oldElem] != -1) //update unchanged element id
        ptclElem(pid) = same_old2New[oldElem];
      else if (modified[oldElem].offset != -1) { //find new split element
        auto key = modified[oldElem].key;

        //Get ptcl position
        Vector<mesh_dim> pos;
        for (int i = 0; i<mesh_dim; i++)
          pos[i] = ptclPos(pid,i);

        //Get parent element
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

        if (side == 1 && are_close(baryCoords[1], 0)){ //case 1: elem on new ptcl
          ptclChild(pid) = 1;
          ptclDim(pid) = 0;
          return; //go to next ptcl
        }
        for (int i=0; i<numEnt[0]; i++) //case 2: elem on old ptcl
          if (are_close(baryCoords[i], 1)) {
            auto newVert = old_vert2new_vert[oldVerts[i]];
            auto firstElemIdx = new_vert2elem.a2ab[newVert];
            ptclElem(pid) = new_vert2elem.ab2b[firstElemIdx];
            ptclChild(pid) = getChildIndex(new_elem2vert, ptclElem(pid), newVert);
            ptclDim(pid) = 0;
            return; //go to next ptcl
          }
        if (side == 1) { //case 3: elem on new edge
          // auto edgeIdx = ptclElem(pid)*nEdges + modified[oldElem].rotation + side;
          // ptclChild(pid) = edgeIdx;
          ptclDim(pid) = 1;
          return; //go to next ptcl
        }
        // for (int i=0; i<numEnt[1]; i++) // case 4: elem on old edge
        //   if (are_close(baryCoords[i], ))
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