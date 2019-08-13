#ifndef GITRM_MESH_HPP
#define GITRM_MESH_HPP
#include <vector>
#include <cfloat>
#include <set>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include "pumipic_adjacency.hpp"
#include "Omega_h_mesh.hpp"

namespace o = Omega_h;
namespace p = pumipic;

//presheath efield is always used. Since it is const, set CONSTANT_EBFIELDS.
// sheath efield is calcualted efield, it is always used. Skip calling 
// gitrm_calculateE for neutrals.

// D3D 0.8 to 2.45 m radial 

constexpr o::LO USE3D_BFIELD = 0; //TOD
constexpr o::LO USE2D_INPUTFIELDS = 0;
// GITR only constant EField is used.
constexpr o::LO USE_CONSTANT_BFIELD = 1; //used for pisces
constexpr o::LO USECYLSYMM = 0; // TODO
constexpr o::LO PISCESRUN  = 1;
constexpr o::Real BACKGROUND_AMU = 4.0; //TODO for pisces
//TODO if multiple species ?
constexpr o::Real PTCL_AMU=184.0; //W

constexpr o::LO BACKGROUND_Z = 1;

constexpr o::Real DEPTH_DIST2_BDRY = 0.001; // 1mm
constexpr o::LO BDRYFACE_SIZE = 30; //TODO
constexpr o::LO BFS_DATA_SIZE = 100;
constexpr o::Real BIAS_POTENTIAL = 250.0;
constexpr o::LO BIASED_SURFACE = 1;
constexpr o::Real CONSTANT_EFIELD[] = {0, 0, 0};
constexpr o::Real CONSTANT_BFIELD[] = {0,0,-0.08};

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


struct FieldStruct2d;
struct FieldStruct3;
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

  /** @brief Fields reals : angle, potential, debyeLength, larmorRadius, 
  *    ChildLangmuirDist
  */
  void initBField(const std::string &, o::Real shiftB=0);
  void parseGridLimits(std::stringstream &, std::string, std::string, bool, 
    bool &, bool &, double &, double &);
  void processFieldFile(const std::string &, o::HostWrite<o::Real> &, 
    FieldStruct2d &, int);
  void load3DFieldOnVtxFromFile(const std::string &, FieldStruct3&, 
    o::Reals&, o::Real shift=0 );

  //TODO delete tags after use/ in destructor
  void addTagAndLoadData(const std::string &, const std::string &);
  void initBoundaryFaces();

  void loadScalarFieldOnBdryFaceFromFile(const std::string &, FieldStruct3 &, 
    o::Real shift=0, int debug=0);
  void load1DFieldOnVtxFromFile(const std::string &, FieldStruct3 &, 
    o::Reals&, o::Reals&, o::Real shift=0, int debug=0);
  
  void markDetectorCylinder(bool render=false);

  void test_interpolateFields(bool debug=false);
  void printDensityTempProfile(double rmax=0.2, int gridsR=20, 
    double zmax=0.5, int gridsZ=10);
  void compareInterpolate2d3d(const o::Reals& data3d, const o::Reals& data2d,
    double x0, double z0, double dx, double dz, int nx, int nz, bool debug=false);

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

struct FieldStruct2d {
  // Implicit call w/o ctr def. not working 
  FieldStruct2d(std::string n, std::string snr, std::string snz, std::string sgr,
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

struct FieldStruct3 {
  FieldStruct3(std::string n, std::string c1, std::string c2, std::string c3,
    std::string g1, std::string g2, std::string g3,
    std::string ng1, std::string ng2, std::string ng3, 
    int nc, int ng, int ngr):
    name(n), comp1(c1), comp2(c2), comp3(c3), 
    grid1str(g1), grid2str(g2), grid3str(g3),
    nGrid1str(ng1), nGrid2str(ng2), nGrid3str(ng3),
    nComp(nc), nGrids(ng), nGridsRead(ngr)
    {}
    ~FieldStruct3() {
      if(data && nComp >0)
        delete data;
      if(grid1 && nGridsRead >0)
        delete grid1;
      if(grid2 && nGridsRead >1)
        delete grid2;
      if(grid3 && nGridsRead >2)
        delete grid3;
    }
  std::string name;
  std::string comp1;
  std::string comp2;
  std::string comp3;
  std::string grid1str;
  std::string grid2str;
  std::string grid3str;
  std::string nGrid1str;
  std::string nGrid2str;
  std::string nGrid3str;
  // 3rd grid not read
  int nComp;
  int nGrids;
  int nGridsRead;
  double gr1Min = 0;
  double gr1Max = 0;
  double gr2Min = 0;
  double gr2Max = 0;
  double gr3Min = 0;
  double gr3Max = 0;
  int nGrid1 = 0;
  int nGrid2 = 0;
  int nGrid3 = 0;
  // All are doubles
  o::HostWrite<o::Real>* data;
  o::HostWrite<o::Real>* grid1;
  o::HostWrite<o::Real>* grid2;
  o::HostWrite<o::Real>* grid3;
};

 // Not used, since this function call from lambda, to modify data, 
 // forces passed argument data to be const
 OMEGA_H_DEVICE void addFaceToBdryData(o::Write<o::Real> &data, 
  o::Write<o::LO> &ids, o::LO fnums, o::LO size, o::LO dof, o::LO fi,
  o::LO fid, o::LO elem, const o::Matrix<3, 3> &face) {
   OMEGA_H_CHECK(fi < fnums);
   for(o::LO i=0; i<size; ++i) {
     for(o::LO j=0; j<3; j++) {
       data[elem*fnums*size + fi*size + i*dof + j] = face[i][j];
     }
   }
   ids[elem*fnums + fi] = fid;
 }

 // Not used; function call from lambda, to change data, forces data to be const
 // Total exposed faces has to be passed in as nbdry; no separate checking
 OMEGA_H_DEVICE void updateAdjElemFlags(const o::LOs &dual_elems,
  const o::LOs &dual_faces, o::LO elem, o::Write<o::LO> &bdryFlags, 
  o::LO nbdry=0) {

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
  if(debug && dataLine)
    std::cout << ":: "<< ind << " : " << ss.str() << " ::1st " << sFirst 
              << " field " << fieldName << "\n";

  if(dataLine) {
    // this is done for every line, not only that of fieldName string
    if(!(sFirst.empty() || sFirst == fieldName)) {
      if(ind < numPtcls || !numPtcls){
        storeData(data, sFirst, ind, iComp, nComp, nans, debug);
      }
    }  
    if(! ss.str().empty()) {
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

inline void processFieldFileFS3(const std::string& fName, FieldStruct3& fs,
  int debug=0) {
  
  std::ifstream ifs(fName);
  if (!ifs.good())
    Omega_h_fail("Error opening Field file %s \n",fName.c_str() );
  // note: grid data are not stored in the main data array

  auto nComp = fs.nComp;
  auto nGrids = fs.nGrids;
  auto nGridsRead = fs.nGridsRead;
  std::string nGridNames[nGrids];
  std::string gridNames[nGrids];
  std::string compNames[nComp];
  bool foundData[nComp], dLine[nComp], foundGrid[nGridsRead], gLine[nGridsRead];
  bool eq=false, dataInit=false;
  int foundGrids=0, indData[nComp], indGrid[nGridsRead];
  std::set<int> nans1, nans2;

  for(int i = 0; i < nComp; ++i) {
    indData[i] = 0;
    foundData[i] = dLine[i] = false;
  }
  for(int i = 0; i < nGridsRead; ++i) {
    indGrid[i] = 0;
    foundGrid[i] = gLine[i] = false;
  }
  nGridNames[0] = fs.nGrid1str;
  if(nGrids>1)
    nGridNames[1] = fs.nGrid2str;
  if(nGrids>2)
    nGridNames[2] = fs.nGrid3str;

  gridNames[0] = fs.grid1str;
  if(nGridsRead>1)
    gridNames[1] = fs.grid2str;
  if(nGridsRead>2)
    gridNames[2] = fs.grid3str;

  compNames[0] = fs.comp1;
  if(nComp>1)
    compNames[1] = fs.comp2;
  if(nComp>2)
    compNames[2] = fs.comp3;

  std::string line, s1, s2, s3;
  while(std::getline(ifs, line)) {
    bool semi = (line.find(';') != std::string::npos);
    std::replace (line.begin(), line.end(), ',' , ' ');
    std::replace (line.begin(), line.end(), ';' , ' ');
    std::stringstream ss(line);
    // first string or number of EACH LINE is got here
    ss >> s1;
    if(s1.find_first_not_of(' ') == std::string::npos) {
      s1 = "";
      if(!semi)
       continue;
    }    
    //grid names
    if(foundGrids < nGrids) {
      for(int i=0; i<nGrids; ++i) {
        if(s1 == nGridNames[i]) {
          ss >> s2 >> s3;
          OMEGA_H_CHECK(s2 == "=");
          int num = std::stoi(s3);
          OMEGA_H_CHECK(!std::isnan(num));
          if(i==0)
            fs.nGrid1 = num;
          else if(i==1)
            fs.nGrid2 = num;
          else if(i==2)
            fs.nGrid3 = num;        
          ++foundGrids;
          if(debug)
            printf("s1 %s %d %d %d \n", s1.c_str(), 
              fs.nGrid1, fs.nGrid2, fs.nGrid3);
        }
      }
    }

    if(!dataInit && foundGrids==nGrids) {
      if(nGridsRead>0)
      fs.grid1 = new o::HostWrite<o::Real>(fs.nGrid1);
      if(nGridsRead>1)
        fs.grid2 = new o::HostWrite<o::Real>(fs.nGrid2);
      if(nGridsRead>2)
        fs.grid3 = new o::HostWrite<o::Real>(fs.nGrid3);

      int ngrid1 = (fs.nGrid1 >0)?fs.nGrid1:1;
      int ngrid2 = (fs.nGrid2 >0)?fs.nGrid2:1;
      int ngrid3 = (fs.nGrid3 >0)?fs.nGrid3:1;
      // data is combined.
      int size = nComp*ngrid1*ngrid2*ngrid3;
      fs.data = new o::HostWrite<o::Real>(size);

      dataInit = true;
    }

    if(dataInit) {
      parseFileFieldData(ss, s1, fs.comp1, semi, *fs.data, indData[0], 
        dLine[0], nans1, eq, 0, nComp, 0, debug>1);
      if(nComp>1)
        parseFileFieldData(ss, s1, fs.comp2, semi, *fs.data, indData[1], 
          dLine[1], nans1, eq, 1, nComp, 0, debug>1);
      if(nComp>2)
        parseFileFieldData(ss, s1, fs.comp3, semi, *fs.data, indData[2], 
          dLine[2], nans1, eq, 2, nComp, 0, debug>1);

      if(nGridsRead>0)
        parseFileFieldData(ss, s1, fs.grid1str, semi, *fs.grid1, indGrid[0], 
         gLine[0], nans2, eq, 0, 1, 0, debug>1);

      if(nGridsRead>1)
        parseFileFieldData(ss, s1, fs.grid2str, semi, *fs.grid2, indGrid[1], 
         gLine[1], nans2, eq, 0, 1, 0, debug>1);

      if(nGridsRead>2)
        parseFileFieldData(ss, s1, fs.grid3str, semi, *fs.grid3, indGrid[2], 
         gLine[2], nans2, eq, 0, 1, 0, debug>1);

      for(int i=0; i<nComp; ++i)
        if(!foundData[i] && dLine[i]) {
          foundData[i] = true;
        }
      for(int i=0; i<nGridsRead; ++i)
        if(!foundGrid[i] && gLine[i]) {
          foundGrid[i] = true;
        }
    }
    s1 = s2 = s3 = "";
  } //while
  OMEGA_H_CHECK(dataInit);

  
  if(debug) {
    printf("ngrids: %d %d : %d %d : %d\n", 
     fs.nGrid1, indGrid[0], fs.nGrid2, indGrid[1], indData[0]);
    if(nComp>1)
      printf("data index2 %d\n", indData[1]);
    if(nComp>2)
      printf("data index3: %d\n", indData[2]);
    if(nGrids==3)
      printf("ngrids3:  %d %d : %d\n", fs.nGrid3, indGrid[2], indData[2]);
  }
  if(nGridsRead>0) {
    fs.gr1Min = (*fs.grid1)[0];
    OMEGA_H_CHECK(fs.nGrid1 == indGrid[0]);
    fs.gr1Max = (*fs.grid1)[fs.nGrid1-1];
  }
  if(nGridsRead>1) {
    fs.gr2Min = (*fs.grid2)[0];
    OMEGA_H_CHECK(fs.nGrid2 == indGrid[1]);
    fs.gr2Max = (*fs.grid2)[fs.nGrid2-1];
  }
  if(nGridsRead>2) {
    fs.gr3Min = (*fs.grid3)[0];
    OMEGA_H_CHECK(fs.nGrid3 == indGrid[2]);
    fs.gr3Max = (*fs.grid3)[fs.nGrid3-1];
  }
  
  
 // for(int i=0; i<nComp; ++i){
    // if ; on first line, dataLine is reset before reaching back
    //OMEGA_H_CHECK(foundData[i]);
 // }
  if(ifs.is_open()) {
    ifs.close();
  }
  if(nans1.size() > 0 || nans2.size() > 0) 
    std::cout << "ERROR: NaN in ADAS file/grid\n";
}

#endif// define
