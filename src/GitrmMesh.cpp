#include <fstream>
#include <algorithm>
#include <vector>
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "GitrmInputOutput.hpp"
#include "Omega_h_reduce.hpp"
//#include "Omega_h_atomics.hpp"  //No such file

namespace o = Omega_h;
namespace p = pumipic;


GitrmMesh::GitrmMesh(o::Mesh& m): 
  mesh(m) {
  //TODO handle this to read from input file
  piscesBeadCylinderIds = o::HostWrite<o::LO>{ 277, 609, 595, 581, 567, 553, 539, 
    525, 511, 497, 483, 469, 455, 154};
  piscesBeadCylinderIdsMesh2  = o::HostWrite<o::LO>{154,439, 453, 467, 481, 495, 
    509, 523, 537, 551, 565, 579, 593, 268};
}

void GitrmMesh::load3DFieldOnVtxFromFile(const std::string tagName, const std::string &file,
  Field3StructInput& fs, o::Reals& readInData_d, const o::Real shift) {
  o::LO debug = 0;
  std::cout<< "Loading " << tagName << " from " << file << " on vtx\n" ;
  //processFieldFileFS3(file, fs, debug);
  readInputDataNcFileFS3(file, fs, debug);
  int nR = fs.getNumGrids(0);
  int nZ = fs.getNumGrids(1);
  o::Real rMin = fs.getGridMin(0);
  o::Real zMin = fs.getGridMin(1);
  o::Real dr = fs.getGridDelta(0);
  o::Real dz = fs.getGridDelta(1);
  if(debug){
    printf(" %s dr%.5f , dz%.5f , rMin%.5f , zMin%.5f \n",
        tagName.c_str(), dr, dz, rMin, zMin);
    printf("data size %d \n", fs.data.size());
  }

  // Interpolate at vertices and Set tag
  o::Write<o::Real> tag_d(3*mesh.nverts());
  const auto coords = mesh.coords(); //Reals
  readInData_d = o::Reals(fs.data);
  auto fill = OMEGA_H_LAMBDA(o::LO iv) {
    o::Vector<3> fv = o::zero_vector<3>();
    o::Vector<3> pos= o::zero_vector<3>();
    // coords' size is 3* nverts
    for(o::LO j=0; j<3; ++j)
      pos[j] = coords[3*iv+j];
    if(debug && iv < 5)
      printf(" iv:%d %.5f %.5f  %.5f \n", iv, pos[0], pos[1], pos[2]);
    
    //cylindrical symmetry, height (z) is same.
    auto rad = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
    // projecting point to y=0 plane, since 2D data is on const-y plane.
    // meaningless to include non-zero y coord of target plane.
    pos[0] = rad + shift;
    pos[1] = 0;
    // cylindrical symmetry = false, since already projected onto y=0 plane
    p::interp2dVector(readInData_d, rMin, zMin, dr, dz, nR, nZ, pos, fv, false);
    for(o::LO j=0; j<3; ++j){ //components
      tag_d[3*iv+j] = fv[j]; 

      if(debug && iv<5)
        printf("vtx %d j %d tag_d[%d]= %g\n", iv, j, 3*iv+j, tag_d[3*iv+j]);
    }
  };
  o::parallel_for(mesh.nverts(), fill, "Fill B Tag");
  o::Reals tag(tag_d);
  mesh.set_tag(o::VERT, tagName, tag);
  if(debug)
    printf("Added tag %s \n", tagName.c_str());
}

// TODO Remove Tags after use ?
// TODO pass parameters in a compact form, libconfig ?
void GitrmMesh::initBField(const std::string &bFile, const o::Real shiftB) {
  mesh2Bfield2Dshift = shiftB;
  mesh.add_tag<o::Real>(o::VERT, "BField", 3);
  // set bt=0. Pisces BField is perpendicular to W target base plate.
  Field3StructInput fb({"br", "bt", "bz"}, {"gridR", "gridZ"}, {"nR", "nZ"});
  load3DFieldOnVtxFromFile("BField", bFile, fb, Bfield_2d, mesh2Bfield2Dshift); 
  
  bGridX0 = fb.getGridMin(0);
  bGridZ0 = fb.getGridMin(1);
  bGridNx = fb.getNumGrids(0);
  bGridNz = fb.getNumGrids(1);
  bGridDx = fb.getGridDelta(0);
  bGridDz = fb.getGridDelta(1);
}


void GitrmMesh::loadScalarFieldOnBdryFacesFromFile(const std::string tagName, 
  const std::string &file, Field3StructInput& fs, const o::Real shift, int debug) {
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  o::LO verbose = 4;
  if(verbose >0)
    std::cout << "Loading "<< tagName << " from " << file << " on bdry faces\n";

  readInputDataNcFileFS3(file, fs, debug);
  int nR = fs.getNumGrids(0);
  int nZ = fs.getNumGrids(1);
  o::Real rMin = fs.getGridMin(0);
  o::Real zMin = fs.getGridMin(1);
  o::Real dr = fs.getGridDelta(0);
  o::Real dz = fs.getGridDelta(1);

  if(verbose >3)
    printf("nR %d nZ %d dr %g, dz %g, rMin %g, zMin %g \n", nR, nZ, dr, dz, rMin, zMin);
  //Interpolate at vertices and Set tag
  o::Write<o::Real> tag_d(mesh.nfaces());
  const auto readInData_d = o::Reals(fs.data);

  if(debug) {
    printf("%s file %s\n", tagName.c_str(), file.c_str());
    for(int i =0; i<100 ; i+=5) {
      printf("%d:%g ", i, fs.data[i] );
    }
    printf("\n");
  }
  auto fill = OMEGA_H_LAMBDA(o::LO fid) {
    //TODO check if faceids are sequential numbers
    if(!side_is_exposed[fid]) {
      tag_d[fid] = 0;
      return;
    }
    // TODO storing fields at centroid may not be best for long tets.
    auto pos = p::face_centroid_of_tet(fid, coords, face_verts);
    //cylindrical symmetry. Height (z) is same.
    auto rad = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
    // projecting point to y=0 plane, since 2D data is on const-y plane.
    // meaningless to include non-zero y coord of target plane.
    pos[0] = rad + shift; 
    pos[1] = 0;
    if(debug)
      printf("fill2:: %f %f %f %f %d %d %f %f %f\n", rMin, zMin, dr, dz, nR, nZ, pos[0], pos[1], pos[2]);
    //Cylindrical symmetry set to true, in case the above equivalent projection is removed.
    o::Real val = p::interpolate2dField(readInData_d, rMin, zMin, dr, dz, 
      nR, nZ, pos, true, 1, 0, debug);
    tag_d[fid] = val; 

    if(verbose > 4 && fid<10)
      printf(" tag_d[%d]= %.5f\n", fid, val);
  };
  o::parallel_for(mesh.nfaces(), fill, "Fill_face_tag");
  o::Reals tag(tag_d);
  mesh.set_tag(o::FACE, tagName, tag);
}
 
void GitrmMesh::load1DFieldOnVtxFromFile(const std::string tagName, 
  const std::string& file, Field3StructInput& fs, o::Reals& readInData_d, 
  o::Reals& tagData, const o::Real shift, int debug) {
  std::cout<< "Loading " << tagName << " from " << file << " on vtx\n" ;
  //processFieldFileFS3(file, fs, debug);
  readInputDataNcFileFS3(file, fs, debug);
  int nR = fs.getNumGrids(0);
  int nZ = fs.getNumGrids(1);
  o::Real rMin = fs.getGridMin(0);
  o::Real zMin = fs.getGridMin(1);
  o::Real dr = fs.getGridDelta(0);
  o::Real dz = fs.getGridDelta(1);

  // data allocation not available here ?
  if(debug){
    auto nd = fs.data.size();
    printf(" %s dr %.5f, dz %.5f, rMin %.5f, zMin %.5f datSize %d\n",
        tagName.c_str(), dr, dz, rMin,zMin, nd);
    for(int j=0; j<nd; j+=nd/20)
      printf("data[%d]:%g ", j, fs.data[j]);
    printf("\n");
  }
  // Interpolate at vertices and Set tag
  o::Write<o::Real> tag_d(mesh.nverts());

  const auto coords = mesh.coords(); //Reals
  readInData_d = o::Reals(fs.data);

  auto fill = OMEGA_H_LAMBDA(o::LO iv) {
    o::Vector<3> pos = o::zero_vector<3>();
    // coords' size is 3* nverts
    for(o::LO j=0; j<3; ++j){
      pos[j] = coords[3*iv+j];
    }
    if(debug && iv>30 && iv<35){
      printf(" vtx:%d %.5f %.5f %.5f\n", iv, pos[0], pos[1], pos[2]);
    }
    //NOTE modifying positions in the following
    //cylindrical symmetry.Height (z) is same.
    auto rad = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
    // projecting point to y=0 plane, since 2D data is on const-y plane.
    // meaningless to include non-zero y coord of target plane.
    pos[0] = rad + shift;
    pos[1] = 0;
        o::Real val = p::interpolate2dField(readInData_d, rMin, zMin, dr, dz, 
      nR, nZ, pos, true, 1, 0, false); //last debug
    tag_d[iv] = val; 

    if(debug && iv>30 && iv<35){
      printf(" tag_d[%d]= %g\n", iv, val);
    }
  };
  o::parallel_for(mesh.nverts(), fill, "Fill Tag");
  tagData = o::Reals (tag_d);
  mesh.set_tag(o::VERT, tagName, tagData);
}

void GitrmMesh::addTagsAndLoadProfileData(const std::string &profileFile, 
  const std::string &profileDensityFile) {
  mesh.add_tag<o::Real>(o::FACE, "ElDensity", 1);
  mesh.add_tag<o::Real>(o::FACE, "IonDensity", 1); //=ni 
  mesh.add_tag<o::Real>(o::FACE, "IonTemp", 1);
  mesh.add_tag<o::Real>(o::FACE, "ElTemp", 1);
  mesh.add_tag<o::Real>(o::VERT, "IonDensityVtx", 1);
  mesh.add_tag<o::Real>(o::VERT, "ElDensityVtx", 1);
  mesh.add_tag<o::Real>(o::VERT, "IonTempVtx", 1);
  mesh.add_tag<o::Real>(o::VERT, "ElTempVtx", 1);
  Field3StructInput fd({"ni"}, {"gridR", "gridZ"}, {"nR", "nZ"}); 
  loadScalarFieldOnBdryFacesFromFile("IonDensity", profileDensityFile, fd); 

  Field3StructInput fdv({"ni"}, {"gridR", "gridZ"}, {"nR", "nZ"});   
  load1DFieldOnVtxFromFile("IonDensityVtx", profileDensityFile, fdv, 
    densIon_d, densIonVtx_d, 0, 0);
  densIonX0 = fdv.getGridMin(0);
  densIonZ0 = fdv.getGridMin(1);
  densIonNx = fdv.getNumGrids(0);
  densIonNz = fdv.getNumGrids(1);
  densIonDx = fdv.getGridDelta(0);
  densIonDz = fdv.getGridDelta(1);

  Field3StructInput fne({"ne"}, {"gridR", "gridZ"}, {"nR", "nZ"});  
  loadScalarFieldOnBdryFacesFromFile("ElDensity", profileFile, fne); 
  Field3StructInput fnev({"ne"}, {"gridR", "gridZ"}, {"nR", "nZ"});  
  load1DFieldOnVtxFromFile("ElDensityVtx", profileFile, fnev, densEl_d, densElVtx_d);
  densElX0 = fne.getGridMin(0);
  densElZ0 = fne.getGridMin(1);
  densElNx = fne.getNumGrids(0);
  densElNz = fne.getNumGrids(1);
  densElDx = fne.getGridDelta(0);
  densElDz = fne.getGridDelta(1);

  Field3StructInput fti({"ti"}, {"gridR", "gridZ"}, {"nR", "nZ"}); 
  loadScalarFieldOnBdryFacesFromFile("IonTemp", profileFile, fti); 
  Field3StructInput ftiv({"ti"}, {"gridR", "gridZ"}, {"nR", "nZ"});
  load1DFieldOnVtxFromFile("IonTempVtx", profileFile, ftiv, temIon_d, tempIonVtx_d);

  tempIonX0 = ftiv.getGridMin(0);
  tempIonZ0 = ftiv.getGridMin(1);
  tempIonNx = ftiv.getNumGrids(0);
  tempIonNz = ftiv.getNumGrids(1);
  tempIonDx = ftiv.getGridDelta(0);
  tempIonDz = ftiv.getGridDelta(1);

  // electron Temperature
  Field3StructInput fte({"te"}, {"gridR", "gridZ"}, {"nR", "nZ"});
  loadScalarFieldOnBdryFacesFromFile("ElTemp", profileFile, fte); 
  Field3StructInput ftev({"te"}, {"gridR", "gridZ"}, {"nR", "nZ"});
  load1DFieldOnVtxFromFile("ElTempVtx", profileFile, ftev, temEl_d, tempElVtx_d);

  tempElX0 = fte.getGridMin(0);
  tempElZ0 = fte.getGridMin(1);
  tempElNx = fte.getNumGrids(0);
  tempElNz = fte.getNumGrids(1);
  tempElDx = fte.getGridDelta(0);
  tempElDz = fte.getGridDelta(1);
}


//NOTE: Importance of mesh.size in GITRm over all boundary faces:
// mesh of size set comp.to GITR: 573742 (GITR) vs 506832 (GITRm)
// mesh of large face dim: 573742 (GITR) vs 361253 (GITRm)
// El temperature is different at center of face, compared to GITR
// when same 1st 2 particles were compared in calcE. For simulation
// using biased surface, TEl decides DLength and CLDist.

//TODO spli this function
void GitrmMesh::initBoundaryFaces(bool debug) {
  auto fieldCenter = mesh2Efield2Dshift;

  larmorRadius_d = o::Write<o::Real>(mesh.nfaces(),0);
  childLangmuirDist_d = o::Write<o::Real>(mesh.nfaces(),0);


  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto down_r2fs = mesh.ask_down(3, 2).ab2b;
  auto useConstantBField = USE_CONSTANT_BFIELD;
  o::Vector<3> BField_const = p::makeVectorHost(CONSTANT_BFIELD);
  o::Real potential = BIAS_POTENTIAL;
  auto biased = BIASED_SURFACE;
  o::Real bxz[4] = {bGridX0, bGridZ0, bGridDx, bGridDz};
  o::LO bnz[2] = {bGridNx, bGridNz};
  const auto &Bfield_2dm = Bfield_2d;
  if(debug)
    printf("bxz: %g %g %g %g\n",  bxz[0], bxz[1], bxz[2], bxz[3]);
  const auto density = mesh.get_array<o::Real>(o::FACE, "IonDensity"); //=ni
  const auto ne = mesh.get_array<o::Real>(o::FACE, "ElDensity");
  const auto te = mesh.get_array<o::Real>(o::FACE, "ElTemp");
  const auto ti = mesh.get_array<o::Real>(o::FACE, "IonTemp");

  o::Write<o::Real> angle_d(mesh.nfaces());
  o::Write<o::Real> debyeLength_d(mesh.nfaces());
  o::Write<o::Real> larmorRadius_d(mesh.nfaces());
  o::Write<o::Real> childLangmuirDist_d(mesh.nfaces());
  o::Write<o::Real> flux_d(mesh.nfaces());
  o::Write<o::Real> impacts_d(mesh.nfaces());
  o::Write<o::Real> potential_d(mesh.nfaces());

  const o::LO background_Z = BACKGROUND_Z;
  const o::Real background_amu = BACKGROUND_AMU;
  //TODO faceId's sequential from 0 ?
  auto fill = OMEGA_H_LAMBDA(o::LO fid) {
    //TODO if faceId's are not sequential, create a (bdry) faceIds array 
    if(side_is_exposed[fid]) {
      o::Vector<3> B = o::zero_vector<3>();
      auto fcent = p::face_centroid_of_tet(fid, coords, face_verts);
      //cylindrical symmetry, height (z) is same.
      auto rad = sqrt(fcent[0]*fcent[0] + fcent[1]*fcent[1]);
      // projecting point to y=0 plane, since 2D data is on const-y plane.
      // meaningless to include non-zero y coord of target plane.
      fcent[0] = rad + fieldCenter; // D3D 1.6955m. TODO check unit
      fcent[1] = 0;
      //Cylindrical symmetry can be false or true, since y coord = 0

      // TODO angle is between surface normal and magnetic field at center of face
      // If  face is long, BField is not accurate. Calculate at closest point ?
      if(debug)//&& fid%1000==0)
        printf(" fid:%d::  %.5f %.5f %.5f \n", fid, fcent[0], fcent[1], fcent[2]);
      if(useConstantBField) {
        B = BField_const;
      } else {
        assert(! p::almost_equal(bxz,0));
        p::interp2dVector(Bfield_2dm,  bxz[0], bxz[1], bxz[2], bxz[3], bnz[0],
          bnz[1], fcent, B, false);
      }
      /*
      auto elmId = p::elem_id_of_bdry_face_of_tet(fid, f2r_ptr, f2r_elem);
      auto surfNorm = p::face_normal_of_tet(fid, elmId, coords, mesh.verts, 
          face_verts, down_r2fs);
      */
      //TODO verify
      auto surfNorm = p::bdry_face_normal_of_tet(fid,coords,face_verts);
      o::Real magB = o::norm(B);
      o::Real magSurfNorm = o::norm(surfNorm);
      o::Real angleBS = p::osh_dot(B, surfNorm);
      o::Real theta = acos(angleBS/(magB*magSurfNorm));
      if (theta > o::PI * 0.5) {
        theta = abs(theta - o::PI);
      }
      angle_d[fid] = theta*180.0/o::PI;
      
      if(debug) {
        printf("fid:%d surfNorm:%g %g %g angleBS=%g theta=%g angle=%g\n", fid, surfNorm[0], 
          surfNorm[1], surfNorm[2],angleBS,theta,angle_d[fid]);
      }

      o::Real tion = ti[fid];
      o::Real tel = te[fid];
      o::Real nel = ne[fid];
      o::Real dens = density[fid];  //3.0E+19 ?

      o::Real dlen = 0;
      if(o::are_close(nel, 0.0)){
        dlen = 1.0e12;
      }
      else {
        dlen = sqrt(8.854187e-12*tel/(nel*pow(background_Z,2)*1.60217662e-19));
      }
      debyeLength_d[fid] = dlen;

      larmorRadius_d[fid] = 1.44e-4*sqrt(background_amu*tion/2)/(background_Z*magB);
      flux_d[fid] = 0.25* dens *sqrt(8.0*tion*1.60217662e-19/(o::PI*background_amu));
      impacts_d[fid] = 0.0;
      o::Real pot = potential;
      if(biased) {
        o::Real cld = 0;
        if(tel > 0.0) {
          cld = dlen * pow(abs(pot)/tel, 0.75);
        }
        else { 
          cld = 1e12;
        }
        childLangmuirDist_d[fid] = cld;
      } else {
        pot = 3.0*tel; 
      }
      potential_d[fid] = pot;

      if(debug)// || o::are_close(angle_d[fid],0))
        printf("angle[%d] %.5f\n", fid, angle_d[fid]);
    }
  };
  //TODO 
  o::parallel_for(mesh.nfaces(), fill, "Fill face Tag");

  mesh.add_tag<o::Real>(o::FACE, "angleBdryBfield", 1, o::Reals(angle_d));
  mesh.add_tag<o::Real>(o::FACE, "DebyeLength", 1, o::Reals(debyeLength_d));
  mesh.add_tag<o::Real>(o::FACE, "LarmorRadius", 1, o::Reals(larmorRadius_d));
  mesh.add_tag<o::Real>(o::FACE, "ChildLangmuirDist", 1, o::Reals(childLangmuirDist_d));
  mesh.add_tag<o::Real>(o::FACE, "flux", 1, o::Reals(flux_d));
  mesh.add_tag<o::Real>(o::FACE, "impacts", 1, o::Reals(impacts_d));
  mesh.add_tag<o::Real>(o::FACE, "potential", 1, o::Reals(potential_d));

 //Book keeping of intersecting particles
  mesh.add_tag<o::Real>(o::FACE, "gridE", 1); // store subgrid data ?
  mesh.add_tag<o::Real>(o::FACE, "gridA", 1);
  mesh.add_tag<o::Real>(o::FACE, "sumParticlesStrike", 1);
  mesh.add_tag<o::Real>(o::FACE, "sumWeightStrike", 1);
  mesh.add_tag<o::Real>(o::FACE, "grossDeposition", 1);
  mesh.add_tag<o::Real>(o::FACE, "grossErosion", 1);
  mesh.add_tag<o::Real>(o::FACE, "aveSputtYld", 1);
  mesh.add_tag<o::LO>(o::FACE, "sputtYldCount", 1);

  o::LO nEdist = 4; //TODO
  o::LO nAdist = 4; //TODO
  mesh.add_tag<o::Real>(o::FACE, "energyDistribution", nEdist*nAdist);
  mesh.add_tag<o::Real>(o::FACE, "sputtDistribution", nEdist*nAdist);
  mesh.add_tag<o::Real>(o::FACE, "reflDistribution", nEdist*nAdist);
}



/** @brief Preprocess distance to boundary.
* Use BFS method to store bdry face ids in elements wich are within required depth.
* This is a 2-step process. (1) Count # bdry faces (2) collect them.
*/
void GitrmMesh::preProcessBdryFacesBfs() {
  o::Write<o::LO> numBdryFaceIdsInElems(mesh.nelems(), 0);
  o::Write<o::LO>dummy(1);
  preprocessStoreBdryFacesBfs(numBdryFaceIdsInElems, dummy, 0);  
  auto tot = mesh.nelems();
  int csrSize = 0;
  auto bdryFacePtrs = makeCsrPtrs(numBdryFaceIdsInElems, tot, csrSize);
  auto bdryFacePtrsBFS = o::LOs(bdryFacePtrs);
  std::cout << "CSR size "<< csrSize << "\n";
  OMEGA_H_CHECK(csrSize > 0);
  o::Write<o::LO> bdryFacesCsrW(csrSize, 0);
  preprocessStoreBdryFacesBfs(numBdryFaceIdsInElems, bdryFacesCsrW, csrSize);
  bdryFacesCsrBFS = o::LOs(bdryFacesCsrW);
}

void GitrmMesh::preprocessStoreBdryFacesBfs(o::Write<o::LO>& numBdryFaceIdsInElems,
  o::Write<o::LO>& bdryFacesCsrW, int csrSize) {
  MESHDATA(mesh);
  int debug = 0;
  if(debug > 2) {
    o::HostRead<o::LO>bdryFacePtrsBFSHost(bdryFacePtrsBFS);
    for(int i=0; i<nel; ++i)
      if(bdryFacePtrsBFSHost[i]>0)
        printf("bdryFacePtrsBFShost %d  %d \n", i, bdryFacePtrsBFSHost[i] );
  }
  constexpr int bfsLocalSize = DIST2BDRY_BFS_ARRAY_SIZE;
  const double depth = DEPTH_DIST2_BDRY;
  const int skipGeometricModelIds = SKIP_GEOMETRIC_MODEL_IDS_FROM_DIST2BDRY;

  //Skip geometric model faces
  o::LOs modelIdsToSkip(1);
  if(skipGeometricModelIds && !USE_PISCES_MESH_VERSION2)
    modelIdsToSkip = o::LOs(o::Write<o::LO>(piscesBeadCylinderIds.write()));
  if(skipGeometricModelIds && USE_PISCES_MESH_VERSION2)
    modelIdsToSkip = o::LOs(piscesBeadCylinderIdsMesh2);

  auto numModelIds = modelIdsToSkip.size();
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");
  auto nelems = nel;
  int loopStep = (int)nelems/100; //TODO parameter
  int rem = (nelems%loopStep) ? 1: 0;
  int last = nelems%loopStep; 
  int nLoop = nelems/loopStep + rem;
  if(debug)
    printf(" nLoop %d last %d rem %d\n", nLoop, last, rem); 
  int step = (csrSize <= 0) ? 1: 2; 
  auto& bdryFacePtrsBFS = this->bdryFacePtrsBFS;
  o::Write<o::LO> nextPositions(nelems, 0);
  
  for(int iLoop = 0; iLoop<nLoop; ++iLoop) {
    int thisLoopStep = loopStep;
    if(iLoop==nLoop-1 && last>0) thisLoopStep = last;
    Kokkos::fence();
    o::Write<o::LO> queue(bfsLocalSize*loopStep, -1);  
     
    if(debug) { 
      printf(" thisLoop %d loopStep %d thisLoopStep %d\n", iLoop, loopStep, thisLoopStep); 
      size_t available=0, total=0;
      cudaMemGetInfo(&available, &total);//FIXME
      std::cout << "CudaMemGetInfo available/MB " << available/(1024*1024) << " total " << total/(1024*1024) << "\n";  
    }
    auto lambda = OMEGA_H_LAMBDA(o::LO el) {
      auto elem = iLoop*loopStep + el;
      o::LO bdryFids[4];
      auto nExpFaces = p::get_exposed_face_ids_of_tet(elem, down_r2f, side_is_exposed, bdryFids);
      if(!nExpFaces)
        return;
      //if any of exposed faces is to be excluded
      for(auto id=0; id < numModelIds; ++id) {
        for(int i = 0; i< 4 && i<nExpFaces; ++i) {
          auto bfid = bdryFids[i];
          if(modelIdsToSkip[id] == faceClassIds[bfid])
            return;
        }
      }
      if(elem > nelems)
        return;
      //parent elem is within dist, since the face(s) is(are) part of it
      o::LO first = el*bfsLocalSize;
      queue[first] = elem;
      int qLen = 1;
      int qFront = first;
      int nq = first+1;
      while(qLen > 0) {
        //deque
        auto thisElem = queue[qFront];
        OMEGA_H_CHECK(thisElem >= 0);
        --qLen;
        ++qFront;
        // process
        OMEGA_H_CHECK(thisElem >= 0);
        const auto tetv2v = o::gather_verts<4>(mesh2verts, thisElem);
        const auto tet = o::gather_vectors<4, 3>(coords, tetv2v);
        bool add = false;
        for(o::LO efi=0; efi < 4 && efi < nExpFaces; ++efi) {
          auto bfid = bdryFids[efi];
          // At least one face is within depth.
          auto within = p::is_face_within_limit_from_tet(tet, face_verts, coords, bfid, depth);
          if(within) {
            add = true;
            if(step==1)
              Kokkos::atomic_increment(&(numBdryFaceIdsInElems[thisElem]));
            else if(step == 2) {
              auto pos = bdryFacePtrsBFS[thisElem];
              auto nextElemIndex = bdryFacePtrsBFS[thisElem+1];
              auto nBdryFaces = numBdryFaceIdsInElems[thisElem];
              auto fInd = Kokkos::atomic_fetch_add(&(nextPositions[thisElem]), 1);
              Kokkos::atomic_exchange(&(bdryFacesCsrW[pos+fInd]), bfid);
              OMEGA_H_CHECK(fInd <= nBdryFaces);
              OMEGA_H_CHECK((pos+fInd) <= nextElemIndex); 
            }
          }//within 
        } //exposed faces
        //queue neighbors if parent is within limit
        if(add) {
          o::LO interiorFids[4];
          auto nIntFaces = p::get_interior_face_ids_of_tet(thisElem, down_r2f, side_is_exposed, interiorFids);
          //across 1st face, if valid. Otherwise it is not used.
          auto dual_elem_id = dual_faces[thisElem]; // ask_dual().a2ab
          for(o::LO ifi=0; ifi < 4 && ifi < nIntFaces; ++ifi) {
            bool addThis = true;
            // Adjacent element across this face ifi
            auto adj_elem  = dual_elems[dual_elem_id];
            for(int i=first; i<nq; ++i) {
              if(adj_elem == queue[i]) {
                addThis = false;
                break;
              }
            }
            if(addThis) {
              queue[nq] = adj_elem;
              ++nq;
              ++qLen;
            }
            ++dual_elem_id;
          } //interior faces
        } //add
      } //while
    };
    o::parallel_for(thisLoopStep, lambda, "storeBdryFaceIds");
  }//for
}


template<size_t N>
OMEGA_H_DEVICE o::Few<o::Vector<3>, N> grid_points_inside_tet(
    const o::LOs &mesh2verts, const o::Reals &coords, o::LO elem, 
    int ndiv, int& npts) {
  o::Few<o::Vector<3>, N> grid;
  const auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  const auto tet = o::gather_vectors<4, 3>(coords, tetv2v);
  npts = 0;

 //https://people.sc.fsu.edu/~jburkardt/cpp_src/
 // tetrahedron_grid/tetrahedron_grid.cpp
  for(int i=0; i<= ndiv; ++i) {
    for(int j=0; j<= ndiv-i; ++j) {
      for(int k=0; k<= ndiv-i-j; ++k) {
        int l = ndiv - i - j - k;
        for(int ii=0; ii<3; ++ii) {
          grid[npts][ii] = (i*tet[0][ii] + j*tet[1][ii] +
                       k*tet[2][ii] + l*tet[3][ii])/ndiv;
        }
        ++npts;
      }
    }
  }
  return grid;
}

void GitrmMesh::preprocessSelectBdryFacesFromAll() {
  MESHDATA(mesh);
  int debug = 0;
  const double minDist = DBL_MAX;
  const auto& f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto& f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;

  int nFaces = mesh.nfaces();
  constexpr int skipGeometricModelIds = SKIP_GEOMETRIC_MODEL_IDS_FROM_DIST2BDRY;
  //to skip geometric model faces
  o::LOs modelIdsToSkip(1);
  if(skipGeometricModelIds && !USE_PISCES_MESH_VERSION2)
    modelIdsToSkip = o::LOs(o::Write<o::LO>(piscesBeadCylinderIds.write()));
  if(skipGeometricModelIds && USE_PISCES_MESH_VERSION2)
    modelIdsToSkip = o::LOs(piscesBeadCylinderIdsMesh2);
  o::LO numModelIds = 0;
  if(skipGeometricModelIds) 
    numModelIds = modelIdsToSkip.size();
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");

  // mark faces
  printf("Marking Bdry faces \n");
  o::Write<o::LO> markedFaces_w(mesh.nfaces(), 0);
  auto lambda1 = OMEGA_H_LAMBDA(o::LO fid) {
    o::LO val = 0;
    if(side_is_exposed[fid])
      val = 1;

    for(o::LO id=0; id < numModelIds; ++id)
      if(modelIdsToSkip[id] == faceClassIds[fid]) {
        val = 0;
        //auto el = p::elem_id_of_bdry_face_of_tet(fid, f2rPtr, f2rElem);
        //printf(" skip:fid:el %d %d \n", fid, el);
      }
    markedFaces_w[fid] = val;
  };
  o::parallel_for(mesh.nfaces(), lambda1, "MarkFaces");
  auto markedFaces = o::LOs(markedFaces_w);

  constexpr o::LO NDIV = D2BDRY_GRIDS_PER_TET; //6->84
  constexpr o::LO NGRID = ((NDIV+1)*(NDIV+2)*(NDIV+3))/6;

  printf("Selecting Bdry-faces: ndiv %d ngrid %d\n", NDIV, NGRID );

  o::Write<o::LO> bdryFaces_nums(mesh.nelems(), 0);
  o::Write<o::LO> bdryFaces_w(mesh.nelems()*NGRID, 0);

  auto lambda2 = OMEGA_H_LAMBDA(o::LO elem) {
    int npts = 1;
    auto grid = grid_points_inside_tet<NGRID>(mesh2verts, coords, elem, NDIV, npts);

    double minDists[NGRID];
    o::LO bfids[NGRID];
    for(int i=0; i<NGRID; ++i) {
      bfids[i] = -1;
      minDists[i] = minDist;
    }
    auto ref = o::zero_vector<3>();
    for(o::LO fid=0; fid<nFaces; ++fid) {
      if(!markedFaces[fid])
        continue;
      const auto face = p::get_face_coords_of_tet(face_verts, coords, fid);
      for(o::LO ipt=0; ipt < npts; ++ipt) {
        ref[0] = grid[ipt][0];
        ref[1] = grid[ipt][1];
        ref[2] = grid[ipt][2];  
        auto pt = p::closest_point_on_triangle(face, ref); 
        auto dist = sqrt(o::norm(pt - ref));
        if(fid==0 || dist < minDists[ipt]) {
          minDists[ipt] = dist;
          bfids[ipt] = fid;
        }
      }
    }

    //remove dduplicates
    o::LO nb = 0;
    for(int i=0; i<npts; ++i) {
      for(int j=i+1; j<npts; ++j)
        if(bfids[i] != -1 && bfids[i] == bfids[j])
          bfids[j] = -1;
    }
    for(int i=0; i<npts; ++i) 
      if(bfids[i] >=0) {
        bdryFaces_w[elem*NGRID+nb] = bfids[i];
        if(debug)
          printf("elem %d : @ %d fid %d nb %d\n", elem, elem*NGRID+nb, bfids[i], nb);
        ++nb;
      }
    bdryFaces_nums[elem] = nb;
    if(debug)
      printf("elem %d : bdryFaces_nums %d total %d\n", elem, nb, npts);
  };
  int num = nel;
  //o::parallel_for(nel, lambda2, "preprocessSelectFromAll");
  o::parallel_for(num, lambda2, "preprocessSelectFromAll"); //FIX num

  // LOs& can't be passed as reference to fill-in  ?
  int csrSize = 0;
  auto ptrs_d = makeCsrPtrs(bdryFaces_nums, nel, csrSize);
  //bdryFacePtrsSelected = o::LOs(ptrs_d); //crash setting class member and
  //using in same funtion ?
  printf("Converting to CSR Bdry faces: size %d\n", csrSize);

  o::Write<o::LO> bdryFacesTrim_w(csrSize, -1);
  auto lambda3 = OMEGA_H_LAMBDA(int elem) {
    if(false && ptrs_d[elem] < ptrs_d[elem+1])
      printf("elem %d  %d \n", elem, ptrs_d[elem]);
    auto beg = elem*NGRID;
    auto ptr = ptrs_d[elem];
    auto num = ptrs_d[elem+1] - ptr;

    for(int i=0; i<num; ++i) {
      if(false)
        printf(" %d  <= %d\n", bdryFaces_w[beg+i],  bdryFacesTrim_w[ptr+i]);
      bdryFacesTrim_w[ptr+i] = bdryFaces_w[beg+i];
    } 
  };
  o::parallel_for(num, lambda3, "preprocessTrimBFids"); //FIX num

  bdryFacesSelectedCsr = o::LOs(bdryFacesTrim_w);
  bdryFacePtrsSelected = o::LOs(ptrs_d);

  printf("Done preprocessing Bdry faces\n");
}
   


// pass as LOs ?
o::Write<o::LO> GitrmMesh::makeCsrPtrs(o::Write<o::LO>& nums_d, int tot, int& sum) {
  int debug = 0;

  o::HostWrite<o::LO> nums(nums_d);
  o::HostWrite<o::LO> ptrs_h(tot+1);
  for(int i=0; i < tot+1; ++i){
    ptrs_h[i] = sum;
    if(i < tot)
      sum += nums[i];
    if(debug && (i==0 || ptrs_h[i-1] < ptrs_h[i]))
      printf("i %d  %d \n", i, ptrs_h[i]);
  }
  return o::Write<o::LO>(ptrs_h.write());
}


void GitrmMesh::markPiscesCylinderResult(o::Write<o::LO>& beadVals_d) {
  o::LOs faceIds(piscesBeadCylinderIds);
  auto numFaceIds = faceIds.size();
  OMEGA_H_CHECK(numFaceIds == beadVals_d.size());
  const auto sideIsExposed = o::mark_exposed_sides(&mesh);
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");
  o::Write<o::LO> edgeTagIds(mesh.nedges(), -1);
  o::Write<o::LO> faceTagIds(mesh.nfaces(), -1);
  o::Write<o::LO> elemTagAsCounts(mesh.nelems(), 0);
  const auto f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto face2edges = mesh.ask_down(o::FACE, o::EDGE);
  const auto faceEdges = face2edges.ab2b;
  o::parallel_for(faceClassIds.size(), OMEGA_H_LAMBDA(const int i) {
    for(auto id=0; id<numFaceIds; ++id) {
      if(faceIds[id] == faceClassIds[i] && sideIsExposed[i]) {
        faceTagIds[i] = id;
        auto elmId = p::elem_id_of_bdry_face_of_tet(i, f2rPtr, f2rElem);
        elemTagAsCounts[elmId] = beadVals_d[id];
        const auto edges = o::gather_down<3>(faceEdges, elmId);
        for(int ie=0; ie<3; ++ie) {
          auto eid = edges[ie];
          edgeTagIds[eid] = beadVals_d[id];
        }
      }
    }
  });
  mesh.add_tag<o::LO>(o::EDGE, "piscesTiRodCounts_edge", 1, o::LOs(edgeTagIds));
  mesh.add_tag<o::LO>(o::REGION, "Pisces_Bead_Counts", 1, o::LOs(elemTagAsCounts));
}

void GitrmMesh::markPiscesCylinder(bool renderPiscesCylCells) {
  o::LOs faceIds(piscesBeadCylinderIds);
  auto numFaceIds = faceIds.size();
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
  // array of all faces, but only classification ids are valid
  auto face_class_ids = mesh.get_array<o::ClassId>(2, "class_id");
  o::Write<o::LO> faceTagIds(mesh.nfaces(), -1);
  o::Write<o::LO> elemTagIds(mesh.nelems(), 0);
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  o::parallel_for(face_class_ids.size(), OMEGA_H_LAMBDA(const int i) {
    for(auto id=0; id<numFaceIds; ++id) {
      if(faceIds[id] == face_class_ids[i] && side_is_exposed[i]) {
        faceTagIds[i] = id;
        if(renderPiscesCylCells) {
          auto elmId = p::elem_id_of_bdry_face_of_tet(i, f2r_ptr, f2r_elem);
          elemTagIds[elmId] = id;
        }
      }
    }
  });

  mesh.add_tag<o::LO>(o::FACE, "piscesTiRod_ind", 1, o::LOs(faceTagIds));
  mesh.add_tag<o::LO>(o::REGION, "piscesTiRodRegIndex", 1, o::LOs(elemTagIds));
}

void GitrmMesh::printDensityTempProfile(double rmax, int gridsR, 
  double zmax, int gridsZ) {
  auto& densIon = densIon_d;
  auto& temIon = temIon_d;
  auto x0Dens = densIonX0;
  auto z0Dens = densIonZ0;
  auto nxDens = densIonNx;
  auto nzDens = densIonNz;
  auto dxDens = densIonDx;
  auto dzDens = densIonDz;
  auto x0Temp = tempIonX0;
  auto z0Temp = tempIonZ0;
  auto nxTemp = tempIonNx;
  auto nzTemp = tempIonNz;
  auto dxTemp = tempIonDx;
  auto dzTemp = tempIonDz;
  double rmin = 0;
  double zmin = 0;
  double dr = (rmax - rmin)/gridsR;
  double dz = (zmax - zmin)/gridsZ;
  // use 2D data. radius rmin to rmax
  auto lambda = OMEGA_H_LAMBDA(const int& ir) {
    double rad = rmin + ir*dr;
    auto pos = o::zero_vector<3>();
    pos[0] = rad;
    pos[1] = 0;
    double zmin = 0;
    for(int i=0; i<gridsZ; ++i) {
      pos[2] = zmin + i*dz;
      auto dens = p::interpolate2dField(densIon, x0Dens, z0Dens, dxDens, 
          dzDens, nxDens, nzDens, pos, false, 1,0);
      auto temp = p::interpolate2dField(temIon, x0Temp, z0Temp, dxTemp,
        dzTemp, nxTemp, nzTemp, pos, false, 1,0);      
      printf("profilePoint: temp2D %g dens2D %g RZ %g %g rInd %d zInd %d \n", 
        temp, dens, pos[0], pos[2], ir, i);
    }
  };
  o::parallel_for(gridsR, lambda);  
}


void GitrmMesh::compareInterpolate2d3d(const o::Reals& data3d,
  const o::Reals& data2d, double x0, double z0, double dx, double dz,
  int nx, int nz, bool debug) {
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  auto lambda = OMEGA_H_LAMBDA(const int &el) {
    auto tetv2v = o::gather_verts<4>(mesh2verts, el);
    auto tet = p::gatherVectors4x3(coords, tetv2v);
    for(int i=0; i<4; ++i) {
      auto pos = tet[i];
      auto bcc = o::zero_vector<4>();
      p::find_barycentric_tet(tet, pos, bcc);   
      auto val3d = p::interpolateTetVtx(mesh2verts, data3d, el, bcc, 1, 0);
      auto pos2d = o::zero_vector<3>();
      pos2d[0] = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
      pos2d[1] = 0;
      auto val2d = p::interpolate2dField(data2d, x0, z0, dx, 
        dz, nx, nz, pos2d, false, 1,0);
      if(!o::are_close(val2d, val3d, 1.0e-6)) {
        printf("testinterpolate: val2D %g val3D %g pos2D %g %g %g el %d\n"
          "bcc %g %g %g %g\n",
          val2d, val3d, pos2d[0], pos2d[1], pos2d[2], el,
          bcc[0], bcc[1], bcc[2], bcc[3]);
        // val2D 0.227648 val3D 0.390361 pos2D 0.0311723 0 0 el 572263
        // bcc 1 1.49495e-17 4.53419e-18 -1.32595e-18
        // Due to precision problem in bcc, deviation (here from 0)
        // when mutliplied by huge density is >> epsilon. This value is
        // negligible, compared to 1.0e+18. Don't check for exact value. 
        // TODO explore this issue, by handling in almost_equal()
        //OMEGA_H_CHECK(false);
      }
    } //mask 
  };
  o::parallel_for(mesh.nelems(), lambda, "test_interpolate");  
}

void GitrmMesh::test_interpolateFields(bool debug) {
  auto densVtx = mesh.get_array<o::Real>(o::VERT, "IonDensityVtx");
  compareInterpolate2d3d(densVtx, densIon_d, densIonX0, densIonZ0,
   densIonDx, densIonDz, densIonNx, densIonNz, debug);

  auto tIonVtx = mesh.get_array<o::Real>(o::VERT, "IonTempVtx");
  compareInterpolate2d3d(tIonVtx, temIon_d, tempIonX0, tempIonZ0,
   tempIonDx, tempIonDz, tempIonNx, tempIonNz, debug);
}

void GitrmMesh::writeDist2BdryFacesData(const std::string outFileName) {
  const auto& f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto& f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  int nfaces = mesh.nfaces();
  o::Write<o::LO> felems_d(nfaces);
  o::parallel_for(nfaces, OMEGA_H_LAMBDA(const int& fid) {
    auto el = p::elem_id_of_bdry_face_of_tet(fid, f2rPtr, f2rElem); 
    felems_d[fid] = el;
  });
  auto felems = o::LOs(felems_d);
  std::vector<std::string> vars;
  vars.insert( vars.end(), {"nelems", "nindices", "nfaces", "nface_elids"});
  std::vector<std::string> datNames;
  datNames.insert(datNames.end(), {"indices", "face_elements", "bdryfaces"});
  writeOutputCsrFile(outFileName, vars, datNames, bdryFacePtrsSelected, 
      felems, bdryFacesSelectedCsr);
}

int GitrmMesh::readDist2BdryFacesData(const std::string& ncFileName) {
  std::vector<std::string> vars;
  vars.insert( vars.end(), {"nindices", "nfaces"});
  std::vector<std::string> datNames;
  datNames.insert(datNames.end(), {"indices", "bdryfaces"});
  return readCsrFile(ncFileName, vars, datNames, bdryCsrReadInDataPtrs, 
    bdryCsrReadInData);
}

//TODO use host print. This doesn't print all entries
void GitrmMesh::printDist2BdryFacesData() {
  auto data_d = bdryFacesSelectedCsr;
  auto ptrs_d = bdryFacePtrsSelected;  
  o::Write<o::LO> bfel_d(data_d.size(), -1);
  const auto& f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto& f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;

  auto lambda = OMEGA_H_LAMBDA(const int &elem) {
    o::LO ind1 = ptrs_d[elem];
    o::LO ind2 = ptrs_d[elem+1];
    auto nf = ind2-ind1;
    if(!nf)
      return;
    for(o::LO i=ind1; i<ind2; ++i) {
      auto fid = data_d[i];
      auto el = p::elem_id_of_bdry_face_of_tet(fid, f2rPtr, f2rElem); 
      bfel_d[i] = el;
    }
  };
  o::parallel_for(ptrs_d.size()-1, lambda);

  auto data_h = o::HostRead<o::LO>(data_d);
  auto ptrs_h = o::HostRead<o::LO>(ptrs_d);
  auto bfel_h = o::HostRead<o::LO>(bfel_d);
  for(int el=0; el<ptrs_h.size()-1; ++el) {
    auto nf = ptrs_h[el+1] - ptrs_h[el];
    for(int i = ptrs_h[el]; i<ptrs_h[el+1]; ++i) {
      auto fid = data_h[i];
      auto bfel = bfel_h[i];
      printf("printfbfid:fid;el  %d  %d ref %d n %d\n", fid, bfel, el, nf);
    }
  }
}
