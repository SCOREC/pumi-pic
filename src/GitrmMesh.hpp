#ifndef GITRM_MESH_HPP
#define GITRM_MESH_HPP

#include "pumipic_adjacency.hpp"

//#include <psTypes.h>
//#include <SellCSigma.h>
//#include <SCS_Macros.h>


namespace o = Omega_h;
namespace p = pumipic;


// The define is good for device too, or pass by template ?
#ifndef DEPTH_DIST2_BDRY
#define DEPTH_DIST2_BDRY 0.001 // 1mm
#endif

#ifndef BDRYFACE_SIZE
#define BDRYFACE_SIZE 100
#endif

// 3 vtx, 1 bdry faceId & 1 bdry elId as Reals
enum { SIZE_PER_FACE = 11, FSKIP=2 };


#define MESHDATA(mesh) \
  const auto nel = mesh.nelems(); \
  const auto coords = mesh.coords(); \
  const auto mesh2verts = mesh.ask_elem_verts(); \
  const auto dual_faces = mesh.ask_dual().ab2b; \
  const auto dual_elems= mesh.ask_dual().a2ab; \
  const auto face_verts = mesh.ask_verts_of(2); \
  const auto down_r2f = mesh.ask_down(3, 2).ab2b; \
  const auto side_is_exposed = mark_exposed_sides(&mesh);

// This class is not a device class since 'this' is not captured by 
// reference in order to access class members within lambdas. 
class GitrmMesh{
public:
  GitrmMesh(o::Mesh &);
  ~GitrmMesh(){}

  // o::Mesh is a reference 
  GitrmMesh(GitrmMesh const&) = delete;
  void operator =(GitrmMesh const&) = delete;

  void initNearBdryDistData();
  void convert2ReadOnlyCSR();
  void copyBdryFacesToSelf();
  void printBdryFaceIds(bool printIds=true, o::LO minNums=0);
  void printBdryFacesCSR(bool printIds=true, o::LO minNums=0);

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
  // Indexes' size is nel+1. Not actual index, but has to x by size of face
  o::LOs bdryFaceInds;
};


//bool gitrm_findDistanceToBdry(const GitrmMesh &gm, 
 //    SellCSigma<Particle>* scs);

 // Not used, since this function call from lambda, to modify data, 
 // forces passed argument data to be const
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

 // Not used; function call from lambda, to change data, forces data to be const
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


#endif// define
