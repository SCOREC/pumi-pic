#ifndef GITRM_MESH_HPP
#define GITRM_MESH_HPP

#include <iostream>

#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"

#include "pumipic_adjacency.hpp"

namespace o = Omega_h;
namespace p = pumipic;

// This class is not a device class since 'this' is not captured by 
// reference in order to access class members within lambdas. 

// The define is good for device too, or pass by template ?
#ifndef DEPTH_DIST2_BDRY
#define DEPTH_DIST2_BDRY 0.001 // 1mm
#endif
#ifndef BDRYFACE_SIZE
#define BDRYFACE_SIZE 500
#endif

// 3 vtx and 1 id as Real
enum { SIZE_PER_FACE = 10 };


#define MESHDATA(mesh) \
  const auto nel = mesh.nelems(); \
  const auto coords = mesh.coords(); \
  const auto mesh2verts = mesh.ask_elem_verts(); \
  const auto dual_faces = mesh.ask_dual().ab2b; \
  const auto dual_elems= mesh.ask_dual().a2ab; \
  const auto face_verts = mesh.ask_verts_of(2); \
  const auto down_r2f = mesh.ask_down(3, 2).ab2b; \
  const auto side_is_exposed = mark_exposed_sides(&mesh);

class GitrmMesh{
public:
  // Pre-process
  GitrmMesh(o::Mesh &);
  ~GitrmMesh(){}

  // o::Mesh is a reference 
  GitrmMesh(GitrmMesh const&) = delete;
  void operator =(GitrmMesh const&) = delete;

  void initNearBdryDistData();
  void convert2ReadOnlyCSR();
  void copyBdryFacesToSelf();
  void printBdryFaceIds(bool, o::LO);

/** Space for a fixed # of Bdry faces is assigned per element, rather than 
   using a common data to be accessed using face id. 
   First stage, only boundary faces are added to the same elements.
   When new faces are added, the owner element updates flags of adj.elements.
   So, the above inital stage sets up flags of adj.eleemnts of bdry face elements. 
   Second stage is updating and passing these faces to second and further adj. levels.
   Each element checks if its flag is set by any adj.element. If so, check all 
   adj. elements for new faces and copies off them.
   Flags are reset before checking adj.elements such that any other thread can still
   set it during copying, in which case the next iteration will be run, even if the 
   data is already copied off in the previous step due to flag/data mismatch.
  */

  void preProcessDistToBdry();
  // Don't make a copy of mesh, or object of this class
  o::Mesh &mesh;
  o::LO numNearBdryElems = 0;
  o::LO numAddedBdryFaces = 0;

  // Simple objects, no data yet
  o::Write<o::Real> bdryFacesW;
  o::Write<o::LO> numBdryFaceIds;
  o::Write<o::LO> bdryFaceIds;
  o::Write<o::LO> bdryFlags;
  o::Write<o::LO> bdryFaceElemIds;
  // convert to Read only CSR after write. Store bdryFaceIds as Real
  // Reals, LOs are const & have const cast of device data
  o::Reals bdryFaces;
  o::LOs bdryFaceInds;
};
//void preProcessDistToBdry(o::Real depth, o::Mesh &mesh, o::Write<o::Real> &bdryFacesW, 
 //   o::Write<o::LO> &numBdryFaceIds, o::Write<o::LO> &bdryFaceIds, o::Write<o::LO> &bdryFlags);
/*void convert2ReadOnlyCSR(o::Mesh &mesh, o::Write<o::Real> &bdryFacesW, o::Write<o::LO> &numBdryFaceIds,
      o::Write<o::LO> &bdryFaceIds, o::Write<o::LO> &bdryFlags, o::Write<o::Real> &bdryFaces, 
      o::Write<o::LO> &bdryFaceInds);
*/
#endif// define

/*
bool gitrm_findDistanceToBdry(const gitrmMesh &gm, 
  SellCSigma<Particle>* scs);
*/

/*
//Not checking if id already in.
OMEGA_H_INLINE void addFaceToBdryData(o::Write<o::Real> &data, o::Write<o::LO> &ids, 
    o::LO fnums, o::LO size, o::LO dof, o::LO fi, o::LO fid, 
    o::LO elem, const o::Matrix<3, 3> &face){
  OMEGA_H_CHECK(fi < fnums);
  //memcpy ?
  for(o::LO i=0; i<size; ++i){
    for(o::LO j=0; j<3; j++){
      data[elem*fnums*size + fi*size + i*dof + j] = face[i][j];
    }
  }
  ids[elem*fnums + fi] = fid;  
}

// flags are const ref ?
// Total exposed faces has to be passed in as nbdry; no separate checking 
OMEGA_H_INLINE void updateAdjElemFlags(const o::LOs &dual_elems, const o::LOs &dual_faces, o::LO elem, 
  o::Write<o::LO> &bdryFlags, o::LO nbdry=0){

  auto dface_ind = dual_elems[elem];
  for(o::LO i=0; i<4-nbdry; ++i){
    auto adj_elem  = dual_faces[dface_ind];
    o::LO val = 1;
    Kokkos::atomic_exchange( &bdryFlags[adj_elem], val);
    ++dface_ind;
  }
}
*/

