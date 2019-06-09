#ifndef GITRM_MESH_HPP
#define GITRM_MESH_HPP

#include "pumipic_adjacency.hpp"
#include "GitrmParticles.hpp"  // For dist2bdry

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


// The define is good for device too, or pass by template ?
static constexpr o::Real  DEPTH_DIST2_BDRY = 0.001; // 1mm
static constexpr o::LO BDRYFACE_SIZE = 100;
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

private:
  //GitrmMesh() = default;
  // static bool hasMesh;
  void calculateCsrIndices(const o::Write<o::LO> &, const bool, o::LOs &, o::LO &);

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
  void parseFileFieldData( std::stringstream &, std::string, 
    std::string, bool, o::HostWrite<o::Real> &, int &, bool &, int, int);
  void parseGridLimits(std::stringstream &, std::string, std::string, bool, 
    bool &, bool &, double &, double &);
  void processFieldFile(const std::string &, o::HostWrite<o::Real> &, 
    FieldStruct &, int);
  void load3DFieldOnVtxFromFile(const std::string &, FieldStruct &);
  void addTagAndLoadData(const std::string &, const std::string &);
  void initBoundaryFaces();

  void loadScalarFieldOnBdryFaceFromFile(const std::string &, FieldStruct &);
  void load1DFieldOnVtxFromFile(const std::string &file, FieldStruct &fs);
    
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
 OMEGA_H_INLINE void addFaceToBdryData(o::Write<o::Real> &data, o::Write<o::LO> &ids,
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


/** @brief Calculate distance of particles to domain boundary 
 * TODO add description of data size of bdryFaces, bdryFaceInds and indexes
 */
inline void gitrm_findDistanceToBdry(  particle_structs::SellCSigma<Particle>* scs,
  o::Mesh &mesh,  const o::Reals &bdryFaces, const o::LOs &bdryFaceInds, 
  const o::LO fsize, const o::LO fskip) {
  const auto nel = mesh.nelems();
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts(); 
  //fskip is 2, since 1st 2 are not part of face vertices
  OMEGA_H_CHECK(fsize > 0 && nel >0);
  auto pos_d = scs->template get<PTCL_POS>();
  auto closestPoint_d = scs->template get<PTCL_BDRY_CLOSEPT>();
  auto bdry_face_d = scs->template get<PTCL_BDRY_FACEID>();
  auto distRun = SCS_LAMBDA(const int &elem, const int &pid,
                                const int &mask){ 
    if (mask > 0) {
    // elem 42200
      o::LO verbose = 1; //(elem%500==0)?2:1;

      o::LO beg = bdryFaceInds[elem];
      o::LO nFaces = bdryFaceInds[elem+1] - beg;
      if(nFaces >0) {
        o::Real dist = 0;
        o::Real min = std::numeric_limits<o::Real>::max();
        o::Few< o::Vector<3>, 3> face;
        o::Vector<3> point{0, 0, 0};
        o::Vector<3> pt;
        o::LO fe = -1;
        o::LO fel = -1;
        o::LO fi = -1;
        o::LO fid = -1;
        o::LO minRegion = -1;

        if(verbose >2)
          printf("\n e_%d nFaces_%d %d %d \n", elem, nFaces, bdryFaceInds[elem], 
            bdryFaceInds[elem+1]);

        for(o::LO ii = 0; ii < nFaces; ++ii) {
          // TODO put in a function
          o::LO ind = (beg + ii)*fsize;
          // fskip=2 for faceId, elId
          OMEGA_H_CHECK(fskip ==2);
          fi = static_cast<o::LO>(bdryFaces[ind]);
          fe = static_cast<o::LO>(bdryFaces[ind + 1]);
          for(o::LO i=0; i<3; ++i) { //Tet vertexes
            for(o::LO j=0; j<3; ++j) { //coords
              face[i][j] = bdryFaces[ind + i*3 + j + fskip];
            }
          }
          auto ref = p::makeVector3(pid, pos_d);
          if(verbose > 2 && ii == 0)
            printf("ref: %d %f %f %f \n", pid, ref[0], ref[1], ref[2]);
          //o::LO region = p::find_closest_point_on_triangle_with_normal(face, ref, point);
          o::LO region = p::find_closest_point_on_triangle(face, ref, pt); 
          dist = p::osh_dot(pt - ref, pt - ref);
          if(verbose >3)
            printf(": dist_%0.6f e_%d reg_%d fe_%d \n", dist, elem, region, fe);

          if(dist < min) {
            min = dist;
            fel = fe;
            fid = fi;
            minRegion = region;
            for(int i=0; i<3; ++i)
              point[0] = pt[i];

            if(verbose >2){
              printf("update:: e_%d dist_%0.6f region_%d fi_%d fe_%d\n", 
                elem, min, region, fi, fe);
              p::print_osh_vector(point, "Nearest_pt");
            }
          }
        }

        min = std::sqrt(min);
        if(verbose >1) {
          printf("  el=%d MINdist=%0.8f fid=%d face_el=%d reg=%d\n", 
            elem, min, fid, fel, minRegion);
        }
        OMEGA_H_CHECK(fid >= 0);
        bdry_face_d(pid) = fid;
        closestPoint_d(pid, 0) = point[0];
        closestPoint_d(pid, 1) = point[1];
        closestPoint_d(pid, 2) = point[2];   
      } 
    }
  };

  scs->parallel_for(distRun);

}

#endif// define
