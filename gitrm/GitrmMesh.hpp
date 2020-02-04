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
 
//TODO put in config class
const int USE_GITR_RND_NUMS = 1;
const bool CREATE_GITR_MESH = false;

const int USE_READIN_CSR_BDRYFACES = 1;
const int WRITE_OUT_BDRY_FACES_FILE = 0;
const bool WRITE_TEXT_D2BDRY_FACES = false;
const bool WRITE_BDRY_FACE_COORDS_NC = false;
const bool WRITE_MESH_FACE_COORDS_NC = false;
//TODO enable runtime
constexpr o::LO D2BDRY_GRIDS_PER_TET = 15;// if csr bdry not re-used

const int USE_2DREADIN_IONI_REC_RATES = 1;
const int USE3D_BFIELD = 0;
const int USE2D_INPUTFIELDS = 1;

// in GITR only constant EField is used.
const int USE_CONSTANT_BFIELD = 1; //used for pisces
const int USE_CYL_SYMMETRY = 1;
const int PISCESRUN  = 1;
const o::LO BACKGROUND_Z = 1;
const o::Real BIAS_POTENTIAL = 250.0;
const o::LO BIASED_SURFACE = 1;
const o::Real CONSTANT_EFIELD0 = 0;
const o::Real CONSTANT_EFIELD1 = 0;
const o::Real CONSTANT_EFIELD2 = 0;
const o::Real CONSTANT_BFIELD0 = 5;
const o::Real CONSTANT_BFIELD1 = 5;
const o::Real CONSTANT_BFIELD2 = -0.08;
// 3 vtx, 1 bdry faceId & 1 bdry elId as Reals. 
enum { BDRY_FACE_STORAGE_SIZE_PER_FACE = 1, BDRY_FACE_STORAGE_IDS=0 };
const int BDRY_STORAGE_SIZE_PER_FACE = 1;
// Elements face type
enum {INTERIOR=1, EXPOSED=2};
const int SKIP_MODEL_IDS_FROM_DIST2BDRY = 0; //set to 0


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
  GitrmMesh(o::Mesh& m);
  //TODO delete tags ?
  ~GitrmMesh(){};

  GitrmMesh(GitrmMesh const&) = delete;
  void operator =(GitrmMesh const&) = delete;

  void createSurfaceGitrMesh(int meshVersion=2, bool markCylFromBdry=true);  
  void printBdryFaceIds(bool printIds=true, o::LO minNums=0);
  void printBdryFacesCSR(bool printIds=true, o::LO minNums=0);
  void test_preProcessDistToBdry();

  o::Mesh& mesh;
  int numDetectorSurfaceFaces = 0;
  o::LO numNearBdryElems = 0;
  o::LO numAddedBdryFaces = 0;

  //GitrmMesh() = default;
  // static bool hasMesh;

  /** Distance to bdry search
  */
  void preProcessBdryFacesBfs();
  o::Write<o::LO> makeCsrPtrs(o::Write<o::LO>& data, int tot, int& sum);
  void preprocessStoreBdryFacesBfs(o::Write<o::LO>& numBdryFaceIdsInElems,
  o::Write<o::LO>& bdryFacesCsrW, int csrSize);

  void writeDist2BdryFacesData(const std::string outFileName="d2bdryFaces.nc", 
    int nD2BdryTetSubDiv=0);
  o::LOs bdryFacesCsrBFS;
  o::LOs bdryFacePtrsBFS;

  void preprocessSelectBdryFacesFromAll(bool init);
  o::LOs bdryFacePtrsSelected;
  o::LOs bdryFacesSelectedCsr;
  void writeBdryFacesDataText(int, std::string fileName="bdryFacesData.txt"); 
  void writeBdryFaceCoordsNcFile(int mode, std::string fileName="meshFaces.nc");
 
  int readDist2BdryFacesData(const std::string &);
  o::LOs bdryCsrReadInDataPtrs;
  o::LOs bdryCsrReadInData;
  
  void initBField(const std::string &, const o::Real shiftB=0);
  void load3DFieldOnVtxFromFile(const std::string, const std::string &,
    Field3StructInput&, o::Reals&, const o::Real shift=0 );
  //TODO delete tags after use/ in destructor
  bool addTagsAndLoadProfileData(const std::string &, const std::string &, const std::string &);
  bool initBoundaryFaces(bool init, bool debug=false);
  void loadScalarFieldOnBdryFacesFromFile(const std::string, const std::string &, 
    Field3StructInput &, const o::Real shift=0, int debug=0);
  void load1DFieldOnVtxFromFile(const std::string, const std::string &, 
    Field3StructInput &, o::Reals&, o::Reals&, const o::Real shift=0, int debug=0);
  int markPiscesCylinder(bool render=false);
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

  //Added for gradient file
  o::Reals gradTi_d;
  //o::Reals gradTiT_d;
  //o::Reals gradTiZ_d;
  o::Reals gradTe_d;
  //o::Reals gradTeT_d;
  //o::Reals gradTeZ_d;
  //Till here

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

  o::Real densElX0 = 0;
  o::Real densElZ0 = 0;
  o::LO densElNx = 0;
  o::LO densElNz = 0;
  o::Real densElDx = 0;
  o::Real densElDz = 0;
  o::Real tempElX0 = 0;
  o::Real tempElZ0 = 0;
  o::LO tempElNx = 0;
  o::LO tempElNz = 0;
  o::Real tempElDx = 0;
  o::Real tempElDz = 0;

  //aDDED FOR GRADIENT FILE
  o::Real gradTiX0 = 0;
  o::Real gradTiZ0 = 0;
  o::Real gradTiNx = 0;
  o::Real gradTiNz = 0;
  o::Real gradTiDx = 0;
  o::Real gradTiDz = 0;


  o::Real gradTeX0 = 0;
  o::Real gradTeZ0 = 0;
  o::Real gradTeNx = 0;
  o::Real gradTeNz = 0;
  o::Real gradTeDx = 0;
  o::Real gradTeDz = 0;


  // till here

  // to replace tag
  o::Reals densIonVtx_d;
  o::Reals tempIonVtx_d;  
  o::Reals densElVtx_d;
  o::Reals tempElVtx_d;
  //added for gradient file 
  o::Reals gradTi_vtx_d;
  //o::Reals gradTiT_vtx_d;
  //o::Reals gradTiZ_vtx_d;
  o::Reals gradTe_vtx_d;
  //o::Reals gradTeT_vtx_d;
  //o::Reals gradTeZ_vtx_d;
  //till here
 
  //get model Ids by opening mesh/model in Simmodeler
  o::HostWrite<o::LO> piscesBeadCylinderIds;
  o::HostWrite<o::LO> modelIdsToSkipFromD2bdry;

  o::Write<o::Real> larmorRadius_d;
  o::Write<o::Real> childLangmuirDist_d;
private:
  bool exists = false;
};

// Cumulative sums. Done on host, to get ordered sum of all previous entries CSR.
// NOTE: numsPerCell must have all entries including zero entries
inline o::LO calculateCsrIndices(const o::LOs& numsPerCell, 
  o::LOs& csrPointers) {
  o::LO tot = numsPerCell.size();
  o::HostRead<o::LO> numsPerCellH(numsPerCell);
  o::HostWrite<o::LO> csrPointersH(tot+1);
  o::Int sum = 0;

  for(o::Int e=0; e < tot+1; ++e){
    csrPointersH[e] = sum; // * S;
    if(e<tot)
      sum += numsPerCellH[e];
  }
  //CSR indices
  csrPointers = o::LOs(csrPointersH.write());
  return sum;
}

#endif// define
