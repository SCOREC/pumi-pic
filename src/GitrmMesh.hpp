#ifndef GITRM_MESH_HPP
#define GITRM_MESH_HPP
#include <vector>
#include <cfloat>
#include <set>
#include <algorithm>
#include <stdexcept>
#include "pumipic_adjacency.hpp"
//#include "GitrmParticles.hpp"  // For dist2bdry
#include "Omega_h_mesh.hpp"

namespace o = Omega_h;
namespace p = pumipic;

#ifndef USE_3D_FIELDS
#define USE_3D_FIELDS 0
#endif

#ifndef BIASED_SURFACE
#define BIASED_SURFACE 0
#endif

// TODO
#ifndef BIAS_POTENTIAL
#define BIAS_POTENTIAL 250.0
#endif

#ifndef USEPRESHEATHEFIELD
#define USEPRESHEATHEFIELD 1
#endif

#ifndef USECYLSYMM
#define USECYLSYMM 0 // TODO
#endif

// protoMPEx/input/gitrInput.cfg
static constexpr o::LO BACKGROUND_Z = 1;
static constexpr o::Real BACKGROUND_AMU = 2.0;
static constexpr o::Real DEPTH_DIST2_BDRY = 0.001; // 1mm
static constexpr o::LO BDRYFACE_SIZE = 20;
static constexpr o::LO BFS_DATA_SIZE = 100;

// 3 vtx, 1 bdry faceId & 1 bdry elId as Reals
enum { SIZE_PER_FACE = 11, FSKIP=2 };

// Elements face type
enum {INTERIOR=1, EXPOSED=2};


#define MESHDATA(mesh) \
  const auto nel = mesh.nelems(); \
  const auto coords = mesh.coords(); \
  const auto mesh2verts = mesh.ask_elem_verts(); \
  const auto dual_faces = mesh.ask_dual().ab2b; \
  const auto dual_elems= mesh.ask_dual().a2ab; \
  const auto face_verts = mesh.ask_verts_of(2); \
  const auto down_r2f = mesh.ask_down(3, 2).ab2b; \
  const auto side_is_exposed = mark_exposed_sides(&mesh);


struct FieldStruct;
class GitrmMesh {
public:
  //TODO make it Singleton; make mesh a pointer, and use function: init(Mesh *mesh) 
  GitrmMesh(o::Mesh &);
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


  //o::Mesh *mesh;
  o::Mesh &mesh;
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

public:
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

public:
  /** @brief Fields reals : angle, potential, debyeLength, larmorRadius, 
  *    ChildLangmuirDist
  */
  void initEandBFields(const std::string &, const std::string &);
  void parseGridLimits(std::stringstream &, std::string, std::string, bool, 
    bool &, bool &, double &, double &);
  void processFieldFile(const std::string &, o::HostWrite<o::Real> &, 
    FieldStruct &, int);
  void load3DFieldOnVtxFromFile(const std::string &, FieldStruct &);

  //TODO delete tags after use/ in destructor
  void addTagAndLoadData(const std::string &, const std::string &);
  void initBoundaryFaces();

  void loadScalarFieldOnBdryFaceFromFile(const std::string &, FieldStruct &);
  void load1DFieldOnVtxFromFile(const std::string &file, FieldStruct &fs);
  
  void markDetectorCylinder(bool render=false);
  //TODO move these to suitable location
  // Used in boundary init and if 2D field is used for particles
  o::Real BGRIDX0 = 0;
  o::Real BGRIDZ0 = 0;
  o::Real BGRID_DX = 0;
  o::Real BGRID_DZ = 0;
  o::LO BGRID_NX = 0;
  o::LO BGRID_NZ = 0;
  o::Real EGRIDX0 = 0;
  o::Real EGRIDZ0 = 0;
  o::Real EGRID_DX = 0;
  o::Real EGRID_DZ = 0;
  o::LO EGRID_NX = 0;
  o::LO EGRID_NZ = 0;
  o::Reals Efield_2d;
  o::Reals Bfield_2d;
};

struct FieldStruct {
  // Implicit call w/o ctr def. not working 
  FieldStruct(std::string n, std::string snr, std::string snz, std::string sgr,
    std::string sgz,std::string sr, std::string st, std::string sz):
    name(n), nrName(snr), nzName(snz), gridR(sgr), gridZ(sgz), rName(sr), 
    tName(st), zName(sz) {}
  std::string name;
  std::string nrName;// "nR"
  std::string nzName; // "nZ"
  std::string gridR;
  std::string gridZ;
  std::string rName;
  std::string tName;
  std::string zName;
  o::Real rMin = 0;
  o::Real rMax = 0;
  o::Real zMin = 0;
  o::Real zMax = 0;
  int nR = 0;
  int nZ = 0;
};

 // Not used, since this function call from lambda, to modify data, 
 // forces passed argument data to be const
 OMEGA_H_DEVICE void addFaceToBdryData(o::Write<o::Real> &data, o::Write<o::LO> &ids,
     o::LO fnums, o::LO size, o::LO dof, o::LO fi, o::LO fid,
     o::LO elem, const o::Matrix<3, 3> &face){
   OMEGA_H_CHECK(fi < fnums);
   for(o::LO i=0; i<size; ++i){
     for(o::LO j=0; j<3; j++){
       data[elem*fnums*size + fi*size + i*dof + j] = face[i][j];
     }
   }
   ids[elem*fnums + fi] = fid;
 }

 // Not used; function call from lambda, to change data, forces data to be const
 // Total exposed faces has to be passed in as nbdry; no separate checking
 OMEGA_H_DEVICE void updateAdjElemFlags(const o::LOs &dual_elems, const o::LOs &dual_faces, o::LO elem,
   o::Write<o::LO> &bdryFlags, o::LO nbdry=0){

   auto dface_ind = dual_elems[elem];
   for(o::LO i=0; i<4-nbdry; ++i){
     auto adj_elem  = dual_faces[dface_ind];
     o::LO val = 1;
     Kokkos::atomic_exchange( &bdryFlags[adj_elem], val);
     ++dface_ind;
   }
 }

// TODO this method is a utility 
// Cumulative sums. Done on host, to get ordered sum of all previous entries CSR.
// NOTE: numsPerSlot must have all entries including zero entries
inline o::LO calculateCsrIndices(const o::LOs& numsPerSlot, o::LOs& csrPointers) {
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

inline void storeData(o::HostWrite<o::Real>& data, std::string sd, int& ind, 
  int iComp, int nComp, std::set<int>& nans, bool debug=false) {
  bool add2nan = false;
  try {
    double num = std::stod(sd);
    if(! std::isnan(num)) { 
      data[nComp*ind+iComp] = num;      
      ++ind;     
    } else 
      add2nan = true;
  } catch (const std::invalid_argument& ia) {
    add2nan = true;
  }
  if(add2nan) {
      nans.insert(ind);
      if(debug)
        std::cout << "WARNING skipping Invalid  input : " << sd 
          << " ind: " << ind << " iComp: " << iComp << "\n";
  }
}

/* file format required:  
  fieldName = num1 num2 ... ;  //for each component, any number of lines 
  line = sFirst + ss, where ss may have '=' if line starts with fieldName.
  semi is ';' to mark end of field, since file may have unused fieldNames
  so that fieldName match can't be the end of previous field.
  numPtcls is max ptcls desired. ind is index of component iComp of fieldName.
  nComp=dof is total components to store in data array.  
*/
inline void parseFileFieldData(std::stringstream& ss, std::string sFirst, 
   std::string fieldName, bool semi, o::HostWrite<o::Real>& data, int& ind,
   bool& dataLine, std::set<int>& nans, bool& expectEqual, int iComp=0, 
   int nComp=1, int numPtcls=0, bool debug=false) {

  if(debug)
    std::cout << ":: " <<  ind << " : " << ss.str() << " " << sFirst << " " << fieldName << "\n";

  std::string s2 = "", sd = "";
  // restart index when string matches
  if(sFirst == fieldName) {
    ind = 0;
    ss >> s2;
    dataLine = true;
    if(s2 != "=")
      expectEqual = true;
    // next character in the same line after fieldNme is '='' 
    if(debug) {
      std::cout << " dataLine: " << dataLine << " of " << fieldName << "\n";
      if(!s2.empty() && s2 != "=")
        std::cout << "WARNING: Unexpected entry: " << s2 << " discarded\n";
    }
  }
  if(dataLine) {
    // this is done for every line, not only that of fieldName string
    if(!(sFirst.empty() || sFirst == fieldName)) {
      if(ind < numPtcls || !numPtcls){
        if(debug)
          std::cout << " storing_first " << sFirst << "\n";
        storeData(data, sFirst, ind, iComp, nComp, nans, debug);
      }
    }  
    if(!ss.str().empty()) {
      while(ss >> sd) {
        if(numPtcls>0 && ind >= numPtcls)
          break;
        // '=' if not with keyword, accept only if first next
        if(expectEqual) {
          expectEqual = false;
          if(sd=="=")
            continue;
        }
        storeData(data, sd, ind, iComp, nComp, nans, debug);
      } 
    }
    if(semi)
      dataLine = false;
  }
}

#endif// define
