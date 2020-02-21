#include <fstream>
#include <algorithm>
#include <vector>
#include <Omega_h_int_scan.hpp>
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "GitrmInputOutput.hpp"
#include "Omega_h_reduce.hpp"
//#include "Omega_h_atomics.hpp"  //No such file

namespace o = Omega_h;
namespace p = pumipic;


GitrmMesh::GitrmMesh(o::Mesh& m): 
  mesh(m) {
  //mesh ht 50cm, rad 10cm. Top lid of small sylinder not included
  detectorSurfaceModelIds = o::HostWrite<o::LO>{268, 593, 579, 565, 551, 537, 
    523, 509, 495,481, 467,453, 439, 154};
  //source materials model ids
  bdryMaterialModelIds = o::HostWrite<o::LO>{138};
  bdryMaterialModelIdsZ = o::HostWrite<o::Real>{74.0};
  //top of inner cylinder and top lid of tower included
  surfaceAndMaterialModelIds = o::HostWrite<o::LO>{268, 593, 579, 565, 
    551, 537, 523, 509, 495,481, 467,453, 439, 154, 150, 138};
  OMEGA_H_CHECK(!exists);
  exists = true;
  setFaceId2BdryFaceIdMap();
  setFaceId2SurfaceAndMaterialIdMap();
  setFaceId2BdryFaceMaterialsZmap();
}

//all boundary faces, ordered
void GitrmMesh::setFaceId2BdryFaceIdMap() {
  auto nf = mesh.nfaces();
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  auto exposed = o::Read<o::I8>(side_is_exposed);
  auto bdryFaces_r = o::offset_scan(exposed);
  auto bdryFaces_d = o::deep_copy(bdryFaces_r);
  o::parallel_for(nf, OMEGA_H_LAMBDA(const o::LO& fid) {
    if(!side_is_exposed[fid])
      bdryFaces_d[fid] = -1;
  });
  bdryFaceOrderedIds = o::LOs(bdryFaces_d);
  nbdryFaces = o::get_sum(exposed);
  OMEGA_H_CHECK(nbdryFaces == (o::HostRead<o::LO>(bdryFaces_r))[nf-1]);
}

void GitrmMesh::setFaceId2SurfaceAndMaterialIdMap() {
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  auto surfaceIds_d = o::LOs(detectorSurfaceModelIds.write());
  auto numSurfIds = surfaceIds_d.size();
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");
  o::Write<o::LO> isSurface_d(mesh.nfaces(), 0);
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {
    if(side_is_exposed[fid]) {
      for(auto id=0; id < numSurfIds; ++id) {
        if(surfaceIds_d[id] == faceClassIds[fid]) {
          isSurface_d[fid] = 1;
        }
      }
    }
  });
  nDetectSurfaces = o::get_sum(o::LOs(isSurface_d));
  auto detSurfIds_r = o::offset_scan(o::LOs(isSurface_d));
  auto detSurfIds_w = o::deep_copy(detSurfIds_r);
  auto nf = mesh.nfaces();
  o::parallel_for(nf, OMEGA_H_LAMBDA(const o::LO& fid) {
    if(!isSurface_d[fid])
      detSurfIds_w[fid]= -1;
  });
  detectorSurfaceOrderedIds = o::LOs(detSurfIds_w);
  OMEGA_H_CHECK(nDetectSurfaces == (o::HostRead<o::LO>(detSurfIds_r))[nf-1]);

  //surface+materials
  auto surfaceAndMaterialIds = o::LOs(surfaceAndMaterialModelIds.write());
  auto numSurfMatIds = surfaceAndMaterialIds.size();
  o::Write<o::LO> isSurfMat_d(mesh.nfaces(), 0);
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {
    if(side_is_exposed[fid]) {
      for(auto id=0; id < numSurfMatIds; ++id) {
        if(surfaceAndMaterialIds[id] == faceClassIds[fid]) {
          isSurfMat_d[fid] = 1;
        }
      }
    }
  });
  nSurfMaterialFaces = o::get_sum(o::LOs(isSurfMat_d));
  auto surfMatIds_r = o::offset_scan(o::LOs(isSurfMat_d));
  auto surfMatIds_w = o::deep_copy(surfMatIds_r);
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {
    if(!isSurfMat_d[fid])
      surfMatIds_w[fid]= -1;
  });
  surfaceAndMaterialOrderedIds = o::LOs(surfMatIds_w);
  OMEGA_H_CHECK(nSurfMaterialFaces == (o::HostRead<o::LO>(surfMatIds_r))[nf-1]);
}

//only material boundary faces
void GitrmMesh::setFaceId2BdryFaceMaterialsZmap() {
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");  
  auto materialIds = o::LOs(bdryMaterialModelIds.write());
  auto matZ_in = o::Reals(bdryMaterialModelIdsZ.write());
  auto numMatIds = bdryMaterialModelIds.size();
  o::Write<o::Real> matZ_d(mesh.nfaces(), 0);
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {
    if(side_is_exposed[fid]) {
      for(auto id=0; id < numMatIds; ++id) {
        if(materialIds[id] == faceClassIds[fid]) {
          matZ_d[fid] = matZ_in[id];
        }
      }
    }
  });
  bdryFaceMaterialZs = o::Reals(matZ_d);
}

//Loads only from 2D input field
void GitrmMesh::load3DFieldOnVtxFromFile(const std::string tagName, 
   const std::string &file, Field3StructInput& fs, o::Reals& readInData_d) {
  o::LO debug = 0;
  std::cout<< "Loading " << tagName << " from " << file << " on vtx\n" ;
  readInputDataNcFileFS3(file, fs, debug);
  int nR = fs.getNumGrids(0);
  int nZ = fs.getNumGrids(1);
  o::Real rMin = fs.getGridMin(0);
  o::Real zMin = fs.getGridMin(1);
  o::Real dr = fs.getGridDelta(0);
  o::Real dz = fs.getGridDelta(1);
  if(debug)
    printf(" %s dr%.5f , dz%.5f , rMin%.5f , zMin%.5f , data-size %d \n",
        tagName.c_str(), dr, dz, rMin, zMin, fs.data.size());

  // Interpolate at vertices and Set tag
  o::Write<o::Real> tag_d(3*mesh.nverts());
  const auto coords = mesh.coords(); //Reals
  readInData_d = o::Reals(fs.data);
  auto fill = OMEGA_H_LAMBDA(const o::LO& iv) {
    o::Vector<3> fv = o::zero_vector<3>();
    o::Vector<3> pos= o::zero_vector<3>();
    // coords' size is 3* nverts
    for(o::LO j=0; j<3; ++j)
      pos[j] = coords[3*iv+j];
    if(debug && iv < 5)
      printf(" iv:%d %.5f %.5f  %.5f \n", iv, pos[0], pos[1], pos[2]);
    bool cylSymm = true;
    p::interp2dVector(readInData_d, rMin, zMin, dr, dz, nR, nZ, pos, fv, cylSymm);
    for(o::LO j=0; j<3; ++j){ //components
      tag_d[3*iv+j] = fv[j]; 

      if(debug && iv<5)
        printf("vtx %d j %d tag_d[%d]= %g\n", iv, j, 3*iv+j, tag_d[3*iv+j]);
    }
  };
  auto fillTagName = "Fill-tag-" + tagName;
  o::parallel_for(mesh.nverts(), fill, fillTagName.c_str());
  o::Reals tag(tag_d);
  mesh.set_tag(o::VERT, tagName, tag);
  if(debug)
    printf("Added tag %s \n", tagName.c_str());
}

void GitrmMesh::initBField(const std::string &bFile) {
  if(USE_CONSTANT_BFIELD) {
    printf("Setting constant BField\n");
    auto bField_h = o::HostWrite<o::Real>({CONSTANT_BFIELD0,
     CONSTANT_BFIELD1, CONSTANT_BFIELD2});
    Bfield_2d = o::Reals(bField_h.write());
    bGridNx = 1;
    bGridNz = 1;
    bGridX0 = 0;
    bGridZ0 = 0;
    bGridDx = 0;
    bGridDz = 0;
  }else {
    mesh.add_tag<o::Real>(o::VERT, "BField", 3);
    // set bt=0. Pisces BField is perpendicular to W target base plate.
    Field3StructInput fb({"br", "bt", "bz"}, {"gridR", "gridZ"}, {"nR", "nZ"});
    load3DFieldOnVtxFromFile("BField", bFile, fb, Bfield_2d); 
    bGridX0 = fb.getGridMin(0);
    bGridZ0 = fb.getGridMin(1);
    bGridNx = fb.getNumGrids(0);
    bGridNz = fb.getNumGrids(1);
    bGridDx = fb.getGridDelta(0);
    bGridDz = fb.getGridDelta(1);
  }
}

//Loads only from 2D input field
void GitrmMesh::loadScalarFieldOnBdryFacesFromFile(const std::string tagName, 
  const std::string &file, Field3StructInput& fs, int debug) {
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  std::cout << "Loading "<< tagName << " from " << file << " on bdry faces\n";
  readInputDataNcFileFS3(file, fs, debug);
  int nR = fs.getNumGrids(0);
  int nZ = fs.getNumGrids(1);
  o::Real rMin = fs.getGridMin(0);
  o::Real zMin = fs.getGridMin(1);
  o::Real dr = fs.getGridDelta(0);
  o::Real dz = fs.getGridDelta(1);

  if(debug)
    printf("nR %d nZ %d dr %g, dz %g, rMin %g, zMin %g \n", 
        nR, nZ, dr, dz, rMin, zMin);
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
  auto fill = OMEGA_H_LAMBDA(const o::LO& fid) {
    if(!side_is_exposed[fid]) {
      tag_d[fid] = 0;
      return;
    }
    auto pos = p::face_centroid_of_tet(fid, coords, face_verts);
    if(debug)
      printf("fill2:: %f %f %f %f %d %d %f %f %f\n", rMin, zMin, dr, dz, 
          nR, nZ, pos[0], pos[1], pos[2]);
    bool cylSymm = true;
    o::Real val = p::interpolate2dField(readInData_d, rMin, zMin, dr, dz, 
      nR, nZ, pos, cylSymm, 1, 0, debug);
    tag_d[fid] = val; 
  };
  o::parallel_for(mesh.nfaces(), fill, "Fill_face_tag");
  o::Reals tag(tag_d);
  mesh.set_tag(o::FACE, tagName, tag);
}
 
//Loads only from 2D input field
void GitrmMesh::load1DFieldOnVtxFromFile(const std::string tagName, 
  const std::string& file, Field3StructInput& fs, o::Reals& readInData_d, 
  o::Reals& tagData, int debug) {
  std::cout<< "Loading " << tagName << " from " << file << " on vtx\n" ;
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

  auto fill = OMEGA_H_LAMBDA(const o::LO& iv) {
    o::Vector<3> pos = o::zero_vector<3>();
    // coords' size is 3* nverts
    for(o::LO j=0; j<3; ++j){
      pos[j] = coords[3*iv+j];
    }
    bool cylSymm = true;
    o::Real val = p::interpolate2dField(readInData_d, rMin, zMin, dr, dz, 
      nR, nZ, pos, cylSymm, 1, 0, false); //last debug
    tag_d[iv] = val; 
  };
  o::parallel_for(mesh.nverts(), fill, "Fill Tag");
  tagData = o::Reals (tag_d);
  mesh.set_tag(o::VERT, tagName, tagData);
}

bool GitrmMesh::addTagsAndLoadProfileData(const std::string &profileFile, 
  const std::string &profileDensityFile, const std::string &profileGradientFile) {
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
    densIon_d, densIonVtx_d);
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

  return true;
}

bool GitrmMesh::initBoundaryFaces(bool init, bool debug) {
  if(!init)
    return false;
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
  o::Reals BField_const(3);
  if(useConstantBField) {
    BField_const = o::Reals(o::HostWrite<o::Real>({CONSTANT_BFIELD0,
      CONSTANT_BFIELD1, CONSTANT_BFIELD2}).write());
  }
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
  const o::Real background_amu = gitrm::BACKGROUND_AMU;
  auto fill = OMEGA_H_LAMBDA(const o::LO& fid) {
    if(side_is_exposed[fid]) {
      o::Vector<3> B = o::zero_vector<3>();
      auto fcent = p::face_centroid_of_tet(fid, coords, face_verts);
      if(debug)
        printf(" fid:%d::  %.5f %.5f %.5f \n", fid, fcent[0], fcent[1], fcent[2]);
      if(useConstantBField) {
        for(auto i=0; i<3; ++i)
          B[i] = BField_const[i];
      } else {
        assert(! p::almost_equal(bxz,0));
        p::interp2dVector(Bfield_2dm,  bxz[0], bxz[1], bxz[2], bxz[3], bnz[0],
          bnz[1], fcent, B, false);
      }
      //normal on boundary points outwards
      auto surfNormOut = p::bdry_face_normal_of_tet(fid,coords,face_verts);
      auto surfNormIn = -surfNormOut;
      o::Real magB = o::norm(B);
      o::Real magSurfNorm = o::norm(surfNormIn);
      o::Real angleBS = o::inner_product(B, surfNormIn);
      o::Real theta = acos(angleBS/(magB*magSurfNorm));
      if (theta > o::PI * 0.5) {
        theta = abs(theta - o::PI);
      }
      angle_d[fid] = theta*180.0/o::PI;
      
      if(debug) {
        printf("fid:%d surfNormOut:%g %g %g angleBS=%g theta=%g angle=%g\n", fid, 
          surfNormOut[0], surfNormOut[1], surfNormOut[2],angleBS,theta,angle_d[fid]);
      }
      o::Real tion = ti[fid];
      o::Real tel = te[fid];
      o::Real nel = ne[fid];
      o::Real dens = density[fid];  //3.0E+19 ?
      o::Real dlen = 0;
      if(o::are_close(nel, 0.0)){
        dlen = 1.0e12;
      }
      else { //TODO use common constants
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
  return true;
}

/** @brief Preprocess distance to boundary.
* Use BFS method to store bdry face ids in elements wich are within required depth.
* This is a 2-step process. (1) Count # bdry faces (2) collect them.
*/
void GitrmMesh::preProcessBdryFacesBfs() {
  auto ne = mesh.nelems();
  o::Write<o::LO> numBdryFaceIdsInElems(ne+1, 0);
  o::Write<o::LO> dummy(1);
  preprocessStoreBdryFacesBfs(numBdryFaceIdsInElems, dummy, 0);  
  int csrSize = 0;
  auto bdryFacePtrs = makeCsrPtrs(numBdryFaceIdsInElems, ne, csrSize);
  auto bdryFacePtrsBFS = o::LOs(bdryFacePtrs);
  std::cout << "CSR size "<< csrSize << "\n";
  OMEGA_H_CHECK(csrSize > 0);
  o::Write<o::LO> bdryFacesCsrW(csrSize, 0);
  preprocessStoreBdryFacesBfs(numBdryFaceIdsInElems, bdryFacesCsrW, csrSize);
  bdryFacesCsrBFS = o::LOs(bdryFacesCsrW);
}

// This method might be needed later
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
  constexpr int bfsLocalSize = 100000;
  const double depth = 0.05;
  constexpr int markFaces = SKIP_MODEL_IDS_FROM_DIST2BDRY;
  o::LOs modelIdsToSkip(1,-1);
  int numModelIds = 0;
  if(markFaces) {
    modelIdsToSkip = o::LOs(detectorSurfaceModelIds);
    numModelIds = modelIdsToSkip.size();
  }
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
    if(debug)
      printf(" thisLoop %d loopStep %d thisLoopStep %d\n", iLoop, loopStep, thisLoopStep); 
    auto lambda = OMEGA_H_LAMBDA(const o::LO& el) {
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
          auto within = p::is_face_within_limit_from_tet(tet, face_verts, 
            coords, bfid, depth);
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
          auto nIntFaces = p::get_interior_face_ids_of_tet(thisElem, 
            down_r2f, side_is_exposed, interiorFids);
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

OMEGA_H_DEVICE int grid_points_inside_tet(const o::LOs& mesh2verts, 
   const o::Reals& coords, const o::LO elem, int ndiv, o::Real* grid) {
  auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
  auto tet = o::gather_vectors<4, 3>(coords, tetv2v);
  int npts = 0;
 //https://people.sc.fsu.edu/~jburkardt/cpp_src/
 // tetrahedron_grid/tetrahedron_grid.cpp
  for(int i=0; i<= ndiv; ++i) {
    for(int j=0; j<= ndiv-i; ++j) {
      for(int k=0; k<= ndiv-i-j; ++k) {
        int l = ndiv - i - j - k;
        for(int ii=0; ii<3; ++ii) {
          grid[npts*3+ii] = (i*tet[0][ii] + j*tet[1][ii] +
                            k*tet[2][ii] + l*tet[3][ii])/ndiv;
        }
        ++npts;
      }
    }
  }
  return npts;
}

//Eliminating using this will cause i/p field dependence
OMEGA_H_DEVICE bool ifBdryAffectsEfieldAt(o::LO fid, const o::Vector<3>& ref, 
   const o::LOs& face_verts, const o::Reals& coords, const o::Real pot,
   const o::Real angle,  const o::Real debyeLen,  const o::Real larmorRad,
   const o::Real childLD, const o::Real emagLimit, const int biasedSurface) { 
  bool debug = 0; 
  auto face = p::get_face_coords_of_tet(face_verts, coords, fid);
  auto pt = p::closest_point_on_triangle(face, ref);
  auto d2bdry = o::norm(pt - ref);
  double emag = 0;
  if(biasedSurface)
    emag = pot/(2.0*childLD)* exp(-d2bdry/(2.0*childLD)); 
  else {
    o::Real fd = 0.98992 + 5.1220E-03 * angle - 7.0040E-04 * pow(angle,2.0) +
                 3.3591E-05 * pow(angle,3.0) - 8.2917E-07 * pow(angle,4.0) +
                 9.5856E-09 * pow(angle,5.0) - 4.2682E-11 * pow(angle,6.0);
    emag = pot*(fd/(2.0 * debyeLen)* exp(-d2bdry/(2.0 * debyeLen))+ 
           (1.0 - fd)/(larmorRad)* exp(-d2bdry/larmorRad));
  }
  return (emag >= emagLimit);
}

void GitrmMesh::preprocessSelectBdryFacesFromAll(bool initBdry) {
  int debug = 0;
  assert(initBdry);
  MESHDATA(mesh);
  const double minDist = DBL_MAX;
  const auto& f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto& f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;

  int nFaces = mesh.nfaces();
  //no skipping
  constexpr int skipGeometricModelIds = SKIP_MODEL_IDS_FROM_DIST2BDRY;
  //to skip geometric model faces
  o::LOs modelIdsToSkip = o::LOs(detectorSurfaceModelIds);
  o::LO numModelIds = 0;
  if(skipGeometricModelIds) 
    numModelIds = modelIdsToSkip.size();
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");

  int biasedSurface = BIASED_SURFACE;
  const auto angles = mesh.get_array<o::Real>(o::FACE, "angleBdryBfield");
  const auto potentials = mesh.get_array<o::Real>(o::FACE, "potential");
  const auto debyeLengths = mesh.get_array<o::Real>(o::FACE, "DebyeLength");
  const auto larmorRadii = mesh.get_array<o::Real>(o::FACE, "LarmorRadius");
  const auto childLDs = mesh.get_array<o::Real>(o::FACE, "ChildLangmuirDist");

  // delete the marking if not needed
  printf("Marking Bdry faces \n");
  o::Write<o::LO> markedFaces_w(mesh.nfaces(), 0);
  auto lambda1 = OMEGA_H_LAMBDA(const o::LO& fid) {
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

  const int ndiv = D2BDRY_GRIDS_PER_TET; //6->84
  const int ngrid = ((ndiv+1)*(ndiv+2)*(ndiv+3))/6;
  printf("Selecting Bdry-faces: ndiv %d ngrid %d\n", ndiv, ngrid);
  const int sizeLimit = 1e9;
  int loopStep = sizeLimit/ngrid; 
  auto nelems = mesh.nelems();
  int rem = (nelems%loopStep) ? 1: 0;
  int last = nelems%loopStep;
  int nLoop = nelems/loopStep + rem;
  if(debug)
     printf(" nLoop %d last %d rem %d\n", nLoop, last, rem);
  //TODO complete this replacement

  o::Write<o::Real> minDists(ngrid, minDist);
  o::Write<o::LO> bfids(ngrid, -1);
  o::Write<o::LO> bdryFaces_nums(mesh.nelems(), 0);
  o::Write<o::LO> bdryFaces_w(mesh.nelems()*ngrid, 0);
  o::Write<o::Real> grid(3*ngrid, 0);
  auto lambda2 = OMEGA_H_LAMBDA(const o::LO& elem) {
    // can't pass Write<> for grid and get filled in ?
    auto npts = grid_points_inside_tet(mesh2verts, coords, elem, ndiv, grid.data());
    for(o::LO fid=0; fid<nFaces; ++fid) {
      if(!markedFaces[fid])
        continue;
      const auto face = p::get_face_coords_of_tet(face_verts, coords, fid);
      for(o::LO ipt=0; ipt < npts; ++ipt) {
        auto ref = o::zero_vector<3>();
        for(auto j=0; j<3; ++j)
          ref[j] = grid[3*ipt+j];
        auto pt = p::closest_point_on_triangle(face, ref); 
        auto dist = o::norm(pt - ref);
        if(fid==0 || dist < minDists[ipt]) {
          minDists[ipt] = dist;
          bfids[ipt] = fid;
        }
      }
    }
    //remove duplicates
    o::LO nb = 0;
    for(int i=0; i<npts; ++i) {
      for(int j=i+1; j<npts; ++j)
        if(bfids[i] != -1 && bfids[i] == bfids[j])
          bfids[j] = -1;
    }
    for(int i=0; i<npts; ++i) {
      auto faceId = bfids[i];
      if(faceId >=0) {
        bool add = true;
        double emagLimit = 0; 
        if(emagLimit >0) {//not used now
          auto point = o::zero_vector<3>();
          for(auto j=0; j<3; ++j)
            point[j] = grid[3*i+j];
          add = ifBdryAffectsEfieldAt(faceId, point, face_verts, coords,
           potentials[faceId], angles[faceId], debyeLengths[faceId],
           larmorRadii[faceId], childLDs[faceId], emagLimit, biasedSurface);
        }
        if(add) {
          bdryFaces_w[elem*ngrid+nb] = faceId;
          ++nb;
        }
        if(false && debug) {
          auto bel = p::elem_id_of_bdry_face_of_tet(faceId,  f2rPtr, f2rElem);
          if(!add)
            printf("skipping-fid %d bel %d from-refel %d\n", faceId, bel, elem);
          printf("adding-fid  %d bel %d : @ %d ref-elem %d nb %d\n", 
            faceId, bel, elem*ngrid+nb, elem, nb);
        }
      }
    }
    bdryFaces_nums[elem] = nb;
    if(debug)
      printf("e %d  nb %d\n", elem, nb);
  };
  o::parallel_for(nel, lambda2, "preprocessSelectFromAll");

  // LOs& can't be passed as reference to fill-in  ?
  int csrSize = 0;
  auto ptrs_d = makeCsrPtrs(bdryFaces_nums, nel, csrSize);
  //bdryFacePtrsSelected = o::LOs(ptrs_d); //crash setting class member and
  //using in same funtion ?
  printf("Converting to CSR Bdry faces: size %d\n", csrSize);

  o::Write<o::LO> bdryFacesTrim_w(csrSize, -1);
  auto lambda3 = OMEGA_H_LAMBDA(const o::LO& elem) {
    if(false && ptrs_d[elem] < ptrs_d[elem+1])
      printf("elem %d  %d \n", elem, ptrs_d[elem]);
    auto beg = elem*ngrid;
    auto ptr = ptrs_d[elem];
    auto size = ptrs_d[elem+1] - ptr;
    for(int i=0; i<size; ++i) {
      if(false)
        printf(" %d  <= %d\n", bdryFaces_w[beg+i],  bdryFacesTrim_w[ptr+i]);
      bdryFacesTrim_w[ptr+i] = bdryFaces_w[beg+i];
    } 
  };
  o::parallel_for(nel, lambda3, "preprocessTrimBFids");
  bdryFacesSelectedCsr = o::LOs(bdryFacesTrim_w);
  bdryFacePtrsSelected = o::LOs(ptrs_d);
  printf("Done preprocessing Bdry faces\n");
}

// nums_d should have size one extra for last ptr
o::Write<o::LO> GitrmMesh::makeCsrPtrs(o::Write<o::LO>& nums_d, int tot, int& sum) {
  auto size =  nums_d.size();
  OMEGA_H_CHECK(tot+1 == size);
  OMEGA_H_CHECK(0 == (o::HostRead<o::LO>(nums_d))[size-1]);
  sum = o::get_sum(o::LOs(nums_d));
  return o::deep_copy(o::offset_scan(o::LOs(nums_d)));
}

void GitrmMesh::writeResultAsMeshTag(o::Write<o::LO>& result_d) {
  //ordered model ids corresp. to  material face indices: 0.. 
  o::LOs faceIds(detectorSurfaceModelIds);
  auto numFaceIds = faceIds.size();
  OMEGA_H_CHECK(numFaceIds == result_d.size());
  const auto sideIsExposed = o::mark_exposed_sides(&mesh);
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");
  o::Write<o::LO> edgeTagIds(mesh.nedges(), -1);
  o::Write<o::LO> faceTagIds(mesh.nfaces(), -1);
  o::Write<o::LO> elemTagAsCounts(mesh.nelems(), 0);
  const auto f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto face2edges = mesh.ask_down(o::FACE, o::EDGE);
  const auto faceEdges = face2edges.ab2b;
  o::parallel_for(faceClassIds.size(), OMEGA_H_LAMBDA(const o::LO& i) {
    for(auto id=0; id<numFaceIds; ++id) {
      if(faceIds[id] == faceClassIds[i] && sideIsExposed[i]) {
        faceTagIds[i] = id;
        auto elmId = p::elem_id_of_bdry_face_of_tet(i, f2rPtr, f2rElem);
        elemTagAsCounts[elmId] = result_d[id];
        const auto edges = o::gather_down<3>(faceEdges, elmId);
        for(int ie=0; ie<3; ++ie) {
          auto eid = edges[ie];
          edgeTagIds[eid] = result_d[id];
        }
      }
    }
  });
  mesh.add_tag<o::LO>(o::EDGE, "Result_edge", 1, o::LOs(edgeTagIds));
  mesh.add_tag<o::LO>(o::REGION, "Result_region", 1, o::LOs(elemTagAsCounts));
}

int GitrmMesh::markDetectorSurfaces(bool render) {
  o::LOs faceIds(detectorSurfaceModelIds);
  auto numFaceIds = faceIds.size();
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
  // array of all faces, but only classification ids are valid
  auto face_class_ids = mesh.get_array<o::ClassId>(2, "class_id");
  o::Write<o::LO> faceTagIds(mesh.nfaces(), -1);
  o::Write<o::LO> elemTagIds(mesh.nelems(), 0);
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  o::Write<o::LO> detFaceCount(1,0);
  o::parallel_for(face_class_ids.size(), OMEGA_H_LAMBDA(const o::LO& i) {
    for(auto id=0; id<numFaceIds; ++id) {
      if(faceIds[id] == face_class_ids[i] && side_is_exposed[i]) {
        faceTagIds[i] = id;
        Kokkos::atomic_increment(&(detFaceCount[0]));
        if(render) {
          auto elmId = p::elem_id_of_bdry_face_of_tet(i, f2r_ptr, f2r_elem);
          elemTagIds[elmId] = id;
        }
      }
    }
  });
  mesh.add_tag<o::LO>(o::FACE, "DetectorSurfaceIndex", 1, o::LOs(faceTagIds));
  mesh.add_tag<o::LO>(o::REGION, "detectorSurfaceRegionInds", 1, o::LOs(elemTagIds));
  auto count_h = o::HostWrite<o::LO>(detFaceCount);
  auto ndf = count_h[0];
  numDetectorSurfaceFaces = ndf;
  return ndf;
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
  auto lambda = OMEGA_H_LAMBDA(const o::LO& ir) {
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
  auto lambda = OMEGA_H_LAMBDA(const o::LO &el) {
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

void GitrmMesh::writeDist2BdryFacesData(const std::string outFileName, int ndiv) {
  int nel = bdryFacePtrsSelected.size() - 1;
  int extra[3];
  extra[0] = nel;
  extra[1] = ndiv; //ordered with those after 2 entries in vars 
  std::vector<std::string> vars{"nindices", "nfaces", "nelems", "nsub_div"};
  std::vector<std::string> datNames{"indices", "bdryfaces"};
  writeOutputCsrFile(outFileName, vars, datNames, bdryFacePtrsSelected, 
      bdryFacesSelectedCsr, extra);
}

int GitrmMesh::readDist2BdryFacesData(const std::string& ncFileName) {
  std::vector<std::string> vars{"nindices", "nfaces"};
  std::vector<std::string> datNames{"indices", "bdryfaces"};
  auto stat = readCsrFile(ncFileName, vars, datNames, bdryCsrReadInDataPtrs,
    bdryCsrReadInData);
  if(stat)
    Omega_h_fail("Error: No Dist2BdryFaces File \n");
  return stat;
}

//mode=1 selected d2bdry faces; 2 all bdry faces
void GitrmMesh::writeBdryFaceCoordsNcFile(int mode, std::string fileName) {
  std::string modestr = (mode==1) ? "d2bdry_stored_" : "allbdry_";
  fileName = modestr + fileName;
  auto readInBdry = USE_READIN_CSR_BDRYFACES;
  o::LOs bdryFaces;
  if(readInBdry)
    bdryFaces = bdryCsrReadInData; 
  else
    bdryFaces = bdryFacesSelectedCsr;
  const auto& face_verts = mesh.ask_verts_of(2);  
  const auto& coords = mesh.coords();
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  auto nFaces = mesh.nfaces();
  auto nbdryFaces = 0;
  Kokkos::parallel_reduce(nFaces, OMEGA_H_LAMBDA(const int i, o::LO& lsum) {
    auto plus = (side_is_exposed[i]) ? 1: 0;
    lsum += plus;
  }, nbdryFaces);
  //current version writes all csr faceids, not checking duplicates
  int nf = nbdryFaces;
  if(mode==1) {
    nf = bdryFaces.size();
    nFaces = nf;
    printf("Writing file %s with csr %d faces\n",
      fileName.c_str(), nf);
  }
  o::Write<o::LO> nextIndex(1,0);
  o::Write<o::Real> bdryx(3*nf,0);
  o::Write<o::Real> bdryy(3*nf,0);
  o::Write<o::Real> bdryz(3*nf,0);
  auto lambda = OMEGA_H_LAMBDA(const o::LO& id) {
    auto fid = id;
    if(mode==2 && !side_is_exposed[fid])
      return;
    if(mode==1)
      fid = bdryFaces[id];
    auto abc = p::get_face_coords_of_tet(face_verts, coords, fid);
    auto ind = Kokkos::atomic_fetch_add(&nextIndex[0], 1);
    if(ind < nf) { //just to use fetch
      for(auto i=0; i<3; ++i) {
        //to be consistent with GITR mesh coords, for vtk conversion
        bdryx[3*ind+i] = abc[i][0];
        bdryy[3*ind+i] = abc[i][1];
        bdryz[3*ind+i] = abc[i][2];
      }
    }
  };
  o::parallel_for(nFaces, lambda);
  writeOutBdryFaceCoordsNcFile(fileName, bdryx, bdryy, bdryz, nf);
} 

void GitrmMesh::writeBdryFacesDataText(int nSubdiv, std::string fileName) {
  auto data_d = bdryFacesSelectedCsr;
  auto ptrs_d = bdryFacePtrsSelected;  
  if(USE_READIN_CSR_BDRYFACES) {
    data_d = bdryCsrReadInData;
    ptrs_d = bdryCsrReadInDataPtrs;
  }
  o::Write<o::LO> bfel_d(data_d.size(), -1);
  const auto& f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto& f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  auto lambda = OMEGA_H_LAMBDA(const o::LO& elem) {
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
   
  std::ofstream outf("Dist2BdryFaces_div"+ std::to_string(nSubdiv)+".txt");
  auto data_h = o::HostRead<o::LO>(data_d);
  auto ptrs_h = o::HostRead<o::LO>(ptrs_d);
  auto bfel_h = o::HostRead<o::LO>(bfel_d);
  for(int el=0; el<ptrs_h.size()-1; ++el) {
    auto nf = ptrs_h[el+1] - ptrs_h[el];
    for(int i = ptrs_h[el]; i<ptrs_h[el+1]; ++i) {
      auto fid = data_h[i];
      auto bfel = bfel_h[i];
      outf << "D2bdry:fid " << fid << " bfel " << bfel << " refel " 
           << el << " nf " << nf << "\n";
    }
  }
  outf.close();
}

// Arr having size() and [] indexing
template<typename Arr>
void outputGitrMeshData(const Arr& data, const o::HostRead<o::Byte>& exposed, 
  const std::vector<std::string>& vars, FILE* fp, std::string format="%.15e") {
  auto dsize = data.size();
  auto nComp = vars.size();
  int len = dsize/nComp;
  for(int comp=0; comp< nComp; ++comp) {
    fprintf(fp, "\n %s = [", vars[comp].c_str());
    int print = 0;
    for(int id=0; id < len; ++id) {
      if(!exposed[id])
        continue;
      if(id >0 && id < len-1)
        fprintf(fp, " , ");
       if(id >0 && id < len-1 && print%5==0)
        fprintf(fp, "\n"); 
      fprintf(fp, format.c_str(), data[id*nComp+comp]);
      ++print; 
    }
    fprintf(fp, " ]\n");
  }
}

void GitrmMesh::createSurfaceGitrMesh() {
  MESHDATA(mesh);
  const auto& f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto& f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  auto nf = mesh.nfaces();
  auto nbdryFaces = 0;
  //count only boundary surfaces
  Kokkos::parallel_reduce(nf, OMEGA_H_LAMBDA(const o::LO& i, o::LO& lsum) {
    auto plus = (side_is_exposed[i]) ? 1: 0;
    lsum += plus;
  }, nbdryFaces);
  printf("Number of boundary faces(including material surfaces) %d total-faces \n", 
    nbdryFaces, nf);
  auto surfaceAndMaterialIds = surfaceAndMaterialOrderedIds;
  //using arrays of all faces, later filtered out
  o::Write<o::Real> points_d(9*nf, 0);
  o::Write<o::Real> abcd_d(4*nf, 0);
  o::Write<o::Real> planeNorm_d(nf, 0);
  o::Write<o::Real> area_d(nf, 0);
  o::Write<o::Real> BCxBA_d(nf, 0);
  o::Write<o::Real> CAxCB_d(nf, 0);
  o::Write<o::LO> surface_d(nf, 0);
  o::Write<o::Real> materialZ_d(nf, 0);
  o::Write<o::LO> inDir_d(nf, -1);
  printf("Creating GITR surface mesh\n");
  auto lambda = OMEGA_H_LAMBDA(const o::LO& fid) {
    if(!side_is_exposed[fid])
      return;
    auto abc = p::get_face_coords_of_tet(face_verts, coords, fid);
    for(auto i=0; i<3; ++i) {
      for(auto j=0; j<3; ++j) {
        points_d[fid*9+i*3+j] = abc[i][j];
      }
    }
    auto ab = abc[1] - abc[0];
    auto ac = abc[2] - abc[0]; 
    auto bc = abc[2] - abc[1];
    auto ba = -ab;
    auto ca = -ac;
    auto cb = -bc;
    auto normalVec = o::cross(ab, ac);
    for(auto i=0; i<3; ++i) {
      abcd_d[fid*4+i] = normalVec[i];
    }
    abcd_d[fid*4+3] = -(o::inner_product(normalVec, abc[0]));
    /*
    //all boundary normals point outwards, still test TODO
    auto elmId = p::elem_id_of_bdry_face_of_tet(fid, f2rPtr, f2rElem);
    //auto surfNormOut = p::bdry_face_normal_of_tet(fid, coords, face_verts);
    auto surfNormOut = p::face_normal_of_tet(fid, elmId, coords, mesh2verts, 
          face_verts, down_r2fs);
    if(p::compare_vector_directions(surfNormOut, normalVec))
      inDir_d[fid] = 1;
    else */
      inDir_d[fid] = -1;
    planeNorm_d[fid] = o::norm(normalVec);
    auto val = o::inner_product(o::cross(bc, ba), normalVec);
    auto sign = (val < 0) ? -1 : 0;
    if(val > 0) sign =1; 
    BCxBA_d[fid] = sign;
    val = o::inner_product(o::cross(ca, cb), normalVec);
    sign = (val < 0) ? -1 : 0;
    if(val > 0) sign = 1;
    CAxCB_d[fid] = sign;
    auto norm1 = o::norm(ab);
    auto norm2 = o::norm(bc);
    auto norm3 = o::norm(ac);
    auto s = (norm1 + norm2 + norm3)/2.0;
    area_d[fid] = sqrt(s*(s-norm1)*(s-norm2)*(s-norm3));
    if(surfaceAndMaterialIds[fid] >= 0)
      surface_d[fid] = 1;
  };
  o::parallel_for(nf, lambda, "fill data");
 
  auto points = o::HostRead<o::Real>(points_d);
  auto abcd = o::HostRead<o::Real>(abcd_d);
  auto planeNorm = o::HostRead<o::Real>(planeNorm_d);
  auto BCxBA = o::HostRead<o::Real>(BCxBA_d);
  auto CAxCB = o::HostRead<o::Real>(CAxCB_d);
  auto area = o::HostRead<o::Real>(area_d);
  auto materialZ_h = o::HostRead<o::Real>(bdryFaceMaterialZs);
  auto surface = o::HostRead<o::LO>(surface_d);
  auto inDir = o::HostRead<o::LO>(inDir_d);
  auto exposed = o::HostRead<o::Byte>(side_is_exposed);

  //geom = \n{ //next x1 = [...] etc comma separated
  //x1,x2,x2,x3,y1,y2,y3,a,b,c,d,plane_norm,BCxBA,CAxCB,area,Z,surface,inDir 
  //periodic = 0; \n theta0 = 0.0; \n theta1 = 0.0; \n} //last separate lines
  std::string fname("gitrGeometryFromGitrm.cfg");  
  FILE* fp = fopen(fname.c_str(), "w");
  fprintf(fp, "geom =\n{\n");  
  std::vector<std::string> strxyz{"x1", "y1", "z1", "x2", "y2", 
                                  "z2", "x3", "y3", "z3"};
  std::string format = "%.15e"; //.5e
  outputGitrMeshData(points, exposed, strxyz, fp, format);
  outputGitrMeshData(abcd, exposed, {"a", "b", "c", "d"}, fp, format);
  outputGitrMeshData(planeNorm, exposed, {"plane_norm"}, fp, format);
  outputGitrMeshData(BCxBA, exposed, {"BCxBA"}, fp, format);
  outputGitrMeshData(CAxCB, exposed, {"CAxCB"}, fp, format);
  outputGitrMeshData(area, exposed, {"area"}, fp, format);
  outputGitrMeshData(materialZ_h, exposed, {"Z"}, fp,"%f");
  outputGitrMeshData(surface, exposed, {"surface"}, fp, "%d");
  outputGitrMeshData(inDir, exposed, {"inDir"}, fp, "%d");
  fprintf(fp, "periodic = 0;\ntheta0 = 0.0;\ntheta1 = 0.0\n}\n"); 
  fclose(fp);
}

