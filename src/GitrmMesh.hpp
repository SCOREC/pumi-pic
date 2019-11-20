#ifndef GITRM_MESH_HPP
#define GITRM_MESH_HPP

#include <vector>
#include <cfloat>
#include <set>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <netcdf>
#include "pumipic_adjacency.hpp"
#include "GitrmInputOutput.hpp"
#include "Omega_h_mesh.hpp"

namespace o = Omega_h;
namespace p = pumipic;

//presheath efield is always used. Since it is const, set CONSTANT_EBFIELDS.
// sheath efield is calcualted efield, it is always used. Skip calling 
// gitrm_calculateE for neutrals.

// D3D 0.8 to 2.45 m radial 

constexpr int COMPARE_WITH_GITR = 1;
constexpr int USE3D_BFIELD = 0; //TODO
constexpr int USE2D_INPUTFIELDS = 0;
// GITR only constant EField is used.
constexpr int USE_CONSTANT_BFIELD = 1; //used for pisces
constexpr int USECYLSYMM = 0; // TODO
constexpr int PISCESRUN  = 1;
constexpr double BACKGROUND_AMU = 4.0; //TODO for pisces
//TODO if multiple species ?
constexpr double PTCL_AMU=184.0; //W
constexpr int BACKGROUND_Z = 1;

  //TODO set to 0 after testing
constexpr int USE_READIN_IONI_REC_RATES = 0;

constexpr double DEPTH_DIST2_BDRY = 0.001; // TODO 0.006; // 1mm
constexpr int BDRYFACE_SIZE = 30 ;// TODO
constexpr int BFS_DATA_SIZE = 100; //100

constexpr double BIAS_POTENTIAL = 250.0;
constexpr int BIASED_SURFACE = 1;
constexpr double CONSTANT_EFIELD[] = {0, 0, 0};
constexpr double CONSTANT_BFIELD[] = {0,0,-0.08};

// 3 vtx, 1 bdry faceId & 1 bdry elId as Reals. 
enum { BDRY_FACE_STORAGE_SIZE_PER_FACE = 1, BDRY_FACE_STORAGE_IDS=2 };
constexpr int BDRY_STORAGE_SIZE_PER_FACE = 1;
// Elements face type
enum {INTERIOR=1, EXPOSED=2};


#define MESHDATA(mesh) \
  const auto nel = mesh.nelems(); \
  const auto coords = mesh.coords(); \
  const auto mesh2verts = mesh.ask_elem_verts(); \
  const auto dual_elems = mesh.ask_dual().ab2b; \
  const auto dual_faces = mesh.ask_dual().a2ab; \
  const auto face_verts = mesh.ask_verts_of(2); \
  const auto down_r2f = mesh.ask_down(3, 2).ab2b; \
  const auto down_r2fs = mesh.ask_down(3, 2).ab2b; \
  const auto side_is_exposed = mark_exposed_sides(&mesh);

class GitrmMesh {
public:
  //TODO make it Singleton; make mesh a pointer, and use function: init(Mesh *mesh) 
  GitrmMesh(o::Mesh& m);
  //TODO delete tags ?
  ~GitrmMesh(){};

  GitrmMesh(GitrmMesh const&) = delete;
  void operator =(GitrmMesh const&) = delete;

  
/** @brief preProcessDistToBdry: Space for a fixed # of Bdry faces is assigned per element.
  * First step: Boundary faces are added to its own element data.
  * When new faces are added, the owner element updates flags of adj.elements.
  * So, the above inital stage sets up flags of adj.eleemnts of bdry face elements. 
  * Second step: updating and passing these faces to adj. elements.
  * Each element checks if its flag is set by any adj. element. If so, check all 
  * adj. elements for new faces and copies off the ids, and data (not tested).
  * Flags are reset before checking adj.elements such that any other thread can still
  * set it during copying, in which case the next iteration will be run, even if the 
  * data is already copied off in the previous step due to flag/data mismatch.
  * @return Flat arrays of bdry face data: bdryFacesW, numBdryFaceIds, 
  * bdryFaceIds, bdryFaceElemIds
  */
  void preProcessDistToBdry();

  void printBdryFaceIds(bool printIds=true, o::LO minNums=0);
  void printBdryFacesCSR(bool printIds=true, o::LO minNums=0);
  void test_preProcessDistToBdry();

  o::Mesh& mesh;
  o::LO numNearBdryElems = 0;
  o::LO numAddedBdryFaces = 0;

  //GitrmMesh() = default;
  // static bool hasMesh;

  void convert2ReadOnlyCSR();
  void copyBdryFacesToSelf();
  void initNearBdryDistData();
  // Data for pre-processing, delete after converting to CSR
  o::Write<o::Real> bdryFacesW;
  o::Write<o::LO> numBdryFaceIds;
  o::Write<o::LO> bdryFaceIds;
  o::Write<o::LO> bdryFlags;
  o::Write<o::LO> bdryFaceElemIds;

  /**  @brief CSR data, used in simulation time-step loop.
  * Storing face data in each element, i.e. same bdry face is copied  
  * to neighboring elements if within the depth. This leads to increased storage,
  * compared to having an array of unique face data, which could be accessed
  * by all elements (slow down).
  * Convert to Read only CSR after write. Store element and bdryFaceIds as Reals. 
  * Reals, LOs are const & have const cast of device data
  */
  o::Reals bdryFaces;
  /** @brief Indexes' size is nel+1. The index has to be mult. by size of face to get
   * array index of data.
   */
  o::LOs bdryFaceInds;

  // Adj search
  void preProcessBdryFacesBFS();
  void preprocessCountBdryFaces(o::Write<o::LO>& numBdryFaceIdsInElems);
  int makeCSRBdryFacePtrs(o::Write<o::LO>& numBdryFaceIdsInElems, o::Read<o::LO>& bdryFacePtrs);
  void preprocessStoreBdryFaces(o::Write<o::LO>& numBdryFaceIdsInElems,
      o::Write<o::Real>& bdryFacesCsrW, o::Read<o::LO>& bdryFacePtrsBFS, int csrSize=0);
  o::Reals bdryFacesCsrBFS;
  o::Read<o::LO> bdryFacePtrsBFS;

  /** @brief Fields reals : angle, potential, debyeLength, larmorRadius, 
  *    ChildLangmuirDist
  */
  void initBField(const std::string &, const o::Real shiftB=0);
  void parseGridLimits(std::stringstream &, std::string, std::string, bool, 
    bool &, bool &, double &, double &);
  void load3DFieldOnVtxFromFile(const std::string, const std::string &,
    Field3StructInput&, o::Reals&, const o::Real shift=0 );
  //TODO delete tags after use/ in destructor
  void addTagsAndLoadProfileData(const std::string &, const std::string &);
  void initBoundaryFaces(bool debug=false);
  void loadScalarFieldOnBdryFacesFromFile(const std::string, const std::string &, 
    Field3StructInput &, const o::Real shift=0, int debug=0);
  void load1DFieldOnVtxFromFile(const std::string, const std::string &, 
    Field3StructInput &, o::Reals&, o::Reals&, const o::Real shift=0, int debug=0);
  void markPiscesCylinder(bool render=false);
  void markPiscesCylinderResult(o::Write<o::LO>& data_d);
  void test_interpolateFields(bool debug=false);
  void printDensityTempProfile(double rmax=0.2, int gridsR=20, 
    double zmax=0.5, int gridsZ=10);
  void compareInterpolate2d3d(const o::Reals& data3d, const o::Reals& data2d,
    double x0, double z0, double dx, double dz, int nx, int nz, bool debug=false);

  std::string profileNcFile = "profile.nc";

  // Used in boundary init and if 2D field is used for particles
  o::Real bGridX0 = 0;
  o::Real bGridZ0 = 0;
  o::Real bGridDx = 0;
  o::Real bGridDz = 0;
  o::LO bGridNx = 0;
  o::LO bGridNz = 0;
  o::Real eGridX0 = 0;
  o::Real eGridZ0 = 0;
  o::Real eGridDx = 0;
  o::Real eGridDz = 0;
  o::LO eGridNx = 0;
  o::LO eGridNz = 0;

  //D3D_major rad =1.6955m; https://github.com/SCOREC/Fusion_Public/blob/master/
  // samples/D-g096333.03337/g096333.03337#L1033
  // field2D center may not coincide with mesh center
  o::Real mesh2Efield2Dshift = 0;
  o::Real mesh2Bfield2Dshift = 0;

  //testing
  o::Reals Efield_2d;
  o::Reals Bfield_2d;
  
  o::Reals densIon_d;
  o::Reals densEl_d;
  o::Reals temIon_d;
  o::Reals temEl_d;
  o::Real densIonX0 = 0;
  o::Real densIonZ0 = 0;
  o::LO densIonNx = 0;
  o::LO densIonNz = 0;
  o::Real densIonDx = 0;
  o::Real densIonDz = 0;
  o::Real tempIonX0 = 0;
  o::Real tempIonZ0 = 0;
  o::LO tempIonNx = 0;
  o::LO tempIonNz = 0;
  o::Real tempIonDx = 0;
  o::Real tempIonDz = 0;

  // to replace tag
  o::Reals densIonVtx_d;
  o::Reals tempIonVtx_d;  
  o::Reals densElVtx_d;
  o::Reals tempElVtx_d;
};

 // Not used, since this function call from lambda, to modify data, 
 // forces passed argument data to be const
 OMEGA_H_DEVICE void addFaceToBdryData(o::Write<o::Real> &data, 
  o::Write<o::LO> &ids, o::LO fnums, o::LO size, o::LO dof, o::LO fi,
  o::LO fid, o::LO elem, const o::Matrix<3, 3> &face) {
   assert(fi < fnums);
   for(o::LO i=0; i<size; ++i) {
     for(o::LO j=0; j<3; j++) {
       data[elem*fnums*size + fi*size + i*dof + j] = face[i][j];
     }
   }
   ids[elem*fnums + fi] = fid;
 }

 // Not used; function call from lambda, to change data, forces data to be const
 // Total exposed faces has to be passed in as nbdry; no separate checking
 OMEGA_H_DEVICE void updateAdjElemFlags(const o::LOs &dual_faces,
  const o::LOs &dual_elems, o::LO elem, o::Write<o::LO> &bdryFlags, 
  o::LO nbdry=0) {

   auto dual_elem_id = dual_faces[elem];
   for(o::LO i=0; i<4-nbdry; ++i){
     auto adj_elem  = dual_elems[dual_elem_id];
     o::LO val = 1;
     Kokkos::atomic_exchange( &bdryFlags[adj_elem], val);
     ++dual_elem_id;
   }
 }

// Cumulative sums. Done on host, to get ordered sum of all previous entries CSR.
// NOTE: numsPerSlot must have all entries including zero entries
inline o::LO calculateCsrIndices(const o::LOs& numsPerSlot, 
  o::LOs& csrPointers) {
  o::LO tot = numsPerSlot.size();
  o::HostRead<o::LO> numsPerSlotH(numsPerSlot);
  o::HostWrite<o::LO> csrPointersH(tot+1);
  o::Int sum = 0;

  for(o::Int e=0; e <= tot; ++e){
    csrPointersH[e] = sum; // * S;
    if(e<tot)
      sum += numsPerSlotH[e];
  }
  //CSR indices
  csrPointers = o::LOs(csrPointersH.write());
  return sum;
}


#endif// define
