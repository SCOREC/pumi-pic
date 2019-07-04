#include <fstream>
#include <algorithm>

#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "Omega_h_reduce.hpp"
//#include "Omega_h_atomics.hpp"  //No such file

namespace o = Omega_h;
namespace p = pumipic;


//o::Mesh* GitrmMesh::mesh = nullptr;
//bool GitrmMesh::hasMesh = false;

GitrmMesh::GitrmMesh(o::Mesh &m): 
  mesh(m) {
}

void GitrmMesh::parseGridLimits(std::stringstream &ss, std::string sfirst, 
  std::string gridName, bool semi, bool &foundMin, bool &gridLine, 
  double &min, double &max){
  
  int verbose = 1;
  std::string s2, s3, sd;
  
  if(sfirst == gridName) {
    ss >> s2 >> s3;
    if(!s3.empty()){
      OMEGA_H_CHECK("=" == s2);
      if(verbose >3) {
        std::cout << " gridLimit:min " << s2 << " " << s3 << "\n";
      }
      min = std::stod(s3);
      foundMin = true;
    }
    else {
      foundMin = false;
    }
    gridLine = true;
  }
  
  if(gridLine) {
    if(!(foundMin || sfirst == gridName)) {
      // s1 is either '=' or number, since grid name already seen.
      if(verbose >3) {
        std::cout << " gridLimit:min: " << sfirst << "\n";
      }      
      min = std::stod(sfirst);
      foundMin = true;
    }
    //TODO Last, only checking line with ';'. If only ';' in last line, this breaks
    if(semi) {
      // If this is the only and last entry in this line
      sd = sfirst;
      //If ss is done, the above sd will be kept
      while(ss >> sd){
      }
      if(verbose >3) {
        std::cout << " gridLimit:max " << sd <<  "\n";
      }
      max = std::stod(sd);
      gridLine = false;

      if(verbose >3) {
        std::cout << "mim,max: " << min << " "<< max << "\n";
      }
    }
    if(verbose >4) {
      std::cout << " ** sfirst " << sfirst << " " << min << " "<< max << "\n";
    }
  }
}



//Depends on netcdf format, and semi colon at the end of fields
void GitrmMesh::processFieldFile(const std::string &fName,
    o::HostWrite<o::Real> &data, FieldStruct &fs, int nComp) {

  o::LO verbose = 1;

  OMEGA_H_CHECK(nComp>0 && nComp<4);
  std::ifstream ifs(fName);
  if (!ifs.is_open()) {
    std::cout << "Error opening Field file " << fName << '\n';
    //exit(1); //TODO
  }

  int ind0 = 0, ind1 = 0, ind2 = 0;
  bool foundMinR, gridLineR, foundMinZ, gridLineZ, dataInit;
  bool dataLine0, dataLine1, dataLine2, doneNr, doneNz;
  bool foundComp0, foundComp1, foundComp2, foundMin0, foundMin1;
  std::string line, s1, s2, s3;

  foundComp0 = foundComp1 = foundComp2 = foundMin0 = foundMin1 = false;
  doneNr = doneNz = dataLine0 = dataLine1 = dataLine2 = false;
  foundMinR = gridLineR = foundMinZ = gridLineZ = dataInit = false;

  while(std::getline(ifs, line)) {
    if(verbose >4)
      std::cout << "Processing  line " << line << '\n';
    // Depends on semicolon at the end of fields
    bool semi = (line.find(';') != std::string::npos);
    std::replace (line.begin(), line.end(), ',' , ' ');
    std::replace (line.begin(), line.end(), ';' , ' ');
    std::stringstream ss(line);
    //First string or number of EACH LINE is got here
    ss >> s1;
    if(verbose >5){
          std::cout << "str s1:" << s1 << "\n";
    }
    // Skip blank line
    if(s1.find_first_not_of(' ') == std::string::npos) {
      s1 = "";
      continue;
    }
    if(s1 == fs.nrName) {
      ss >> s2 >> s3;
      fs.nR = std::stoi(s3);
      doneNr = true;
    }
    else if(s1 == fs.nzName) {
      ss >> s2 >> s3;
      fs.nZ = std::stoi(s3);
      doneNz = true;
    }
    
    if(!dataInit && doneNr && doneNz) {
      data = o::HostWrite<o::Real>(nComp*fs.nR*fs.nZ); //destruct ?
      dataInit = true;
      if(verbose >3) {
        std::cout << " nR,nZ: " << fs.nR << " "<< fs.nZ << " \n";
      }
    }

    parseGridLimits(ss, s1, fs.gridR, semi, foundMinR, gridLineR, fs.rMin, fs.rMax);
    if(!foundMin0 && foundMinR) {
      foundMin0 = true;
    } 
    parseGridLimits(ss, s1, fs.gridZ, semi, foundMinZ, gridLineZ, fs.zMin, fs.zMax);
    if(!foundMin1 && foundMinZ) {
      foundMin1 = true;
    }
    if(dataInit) {
      parseFileFieldData(ss, s1, fs.rName, semi, data, ind0, dataLine0, 0, nComp);
    } 
    if(!foundComp0 && dataLine0) {
      foundComp0 = true;
    } 
    // 2nd component
    if(nComp >1) {
      if(dataInit){
        parseFileFieldData(ss, s1, fs.zName, semi, data, ind1, dataLine1, 1, nComp);
      }
      if(!foundComp1 && dataLine1) {
        foundComp1 = true;
      } 
    }
    // 3rd component
    if(nComp >2) {
      if(dataInit){
        parseFileFieldData(ss, s1, fs.tName, semi, data, ind2, dataLine2, 2, nComp);
      }
      if(!foundComp2 && dataLine2) {
        foundComp2 = true;
      } 
    }

    s1 = s2 = s3 = "";
  } //while

  if(verbose >3){
    std::cout << "foundMin0 : " << foundMin0  << 
      " foundMin1 : " << foundMin1 << "\n";
  }
  OMEGA_H_CHECK(foundMin0 && foundMin1);
  OMEGA_H_CHECK(dataInit);

  OMEGA_H_CHECK(foundComp0);
  if(nComp > 1){
    if(verbose >2){
      std::cout << " fs.zName: " << foundComp1 << "\n";
    }
    OMEGA_H_CHECK(foundComp0 && foundComp1);
  }
  if(nComp > 2){
    if(verbose >2){
      std::cout << " fs.tName: " << foundComp2 << "\n";
    }
    OMEGA_H_CHECK(foundComp0 && foundComp1 && foundComp2);
  }

  if(verbose >2){
    std::cout << "Found component " << fs.rName << " = " << foundComp0  << ":: " 
              << fs.nR << " " << fs.nZ << "\n";
  }

  if(ifs.is_open()) {
    ifs.close();
  }
}


void GitrmMesh::load3DFieldOnVtxFromFile(const std::string &file, FieldStruct &fs) {
  o::LO verbose = 1;

  std::cout<< "Loading " << fs.name << " from " << file << " on vtx\n" ;
  // Not per vertex; but for input EField: [nZ*nR]
  o::HostWrite<o::Real> readInData;

  processFieldFile(file, readInData, fs, 3);
  std::string tagName = fs.name;
  o::Real rMin = fs.rMin;
  o::Real rMax = fs.rMax;
  o::Real zMin = fs.zMin;
  o::Real zMax = fs.zMax;
  int nR = fs.nR;
  int nZ = fs.nZ;
  OMEGA_H_CHECK(fs.nR >0 && fs.nZ >0);
  o::Real dr = (rMax - rMin)/nR;
  o::Real dz = (zMax - zMin)/nZ;
  if(verbose >2){
    printf(" dr%.5f , dz%.5f , rMax%.5f , rMin%.5f , zMax%.5f, zMin%.5f \n",
        dr, dz, rMax, rMin,zMax , zMin);
  }

  // Interpolate at vertices and Set tag
  o::Write<o::Real> tag_d(3*mesh.nverts());

  const auto coords = mesh.coords(); //Reals
  const auto readInData_d = o::Reals(readInData);

  // temporary
  if(fs.name == "BField")
    Bfield_2d = readInData_d;
  if(fs.name == "EField")
    Efield_2d = readInData_d;

  auto fill = OMEGA_H_LAMBDA(o::LO iv) {
    o::Vector<3> fv = o::zero_vector<3>();
    o::Vector<3> pos= o::zero_vector<3>();
    // coords' size is 3* nverts
    for(o::LO j=0; j<3; ++j){
      pos[j] = coords[3*iv+j];
      if(verbose > 3 && iv<5){
        printf(" iv:%d %.5f \n", iv, pos[j]);
      }
    }
    //TODO Cylindrical symmetry = true
    p::interp2dVector(readInData_d, rMin, zMin, dr, dz, nR, nZ, pos, fv, true);
    for(o::LO j=0; j<3; ++j){ //components
      tag_d[3*iv+j] = fv[j]; 

      if(verbose > 3 && iv<10){
        printf(" tag_d[%d]= %.5f\n", 3*iv+j, tag_d[3*iv+j]);
      }
    }

  };
  o::parallel_for(mesh.nverts(), fill, "Fill E/B Tag");
  o::Reals tag(tag_d);
  mesh.set_tag(o::VERT, tagName, tag);
}

// TODO Remove Tags after use
// TODO pass parameters in a compact form, libconfig ?
void GitrmMesh::initEandBFields(const std::string &bFile, const std::string &eFile) {
  mesh.add_tag<o::Real>(o::VERT, "BField", 3);
  mesh.add_tag<o::Real>(o::VERT, "EField", 3);

  // Load BField
  FieldStruct fb{"BField", "nR", "nZ", "r", "z", "br", "bt", "bz"};
  load3DFieldOnVtxFromFile(bFile, fb); 

  //TODO this is not a good way
  BGRIDX0 = fb.rMin;
  BGRIDZ0 = fb.zMin;
  BGRID_NX = fb.nR;
  BGRID_NZ = fb.nZ;
  BGRID_DX = (fb.rMax - fb.rMin)/fb.nR;
  BGRID_DZ = (fb.zMax - fb.zMin)/fb.nZ;

  // Load EField
  FieldStruct fel{"EField", "nR", "nZ", "gridR", "gridZ", "er", "et", "ez"};
  load3DFieldOnVtxFromFile(eFile, fel);
 //TODO this is not a good way
  EGRIDX0 = fel.rMin;
  EGRIDZ0 = fel.zMin;
  EGRID_NX = fel.nR;
  EGRID_NZ = fel.nZ;
  EGRID_DX = (fel.rMax - fel.rMin)/fel.nR;
  EGRID_DZ = (fel.zMax - fel.zMin)/fel.nZ; 
}


void GitrmMesh::loadScalarFieldOnBdryFaceFromFile(const std::string &file, 
  FieldStruct &fs) {
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  o::LO verbose = 1;
  if(verbose >0)
    std::cout << "Loading "<< fs.name << " from " << file << " on bdry\n";

  o::HostWrite<o::Real> readInData;

  processFieldFile(file, readInData, fs, 1); // 1= nComp
  std::string tagName = fs.name;
  o::Real rMin = fs.rMin;
  o::Real rMax = fs.rMax;
  o::Real zMin = fs.zMin;
  o::Real zMax = fs.zMax;
  int nR = fs.nR;
  int nZ = fs.nZ;
  OMEGA_H_CHECK(fs.nR >0 && fs.nZ >0);
  o::Real dr = (rMax - rMin)/nR;
  o::Real dz = (zMax - zMin)/nZ;
  if(verbose >3){
    printf(" dr%.5f , dz%.5f , rMax%.5f , rMin%.5f , zMax%.5f, zMin%.5f \n",
        dr, dz, rMax, rMin,zMax , zMin);
  }
  //Interpolate at vertices and Set tag
  //auto tag =  o::Write<o::Real>(mesh.nverts()*3, 0);
  o::Write<o::Real> tag_d(mesh.nfaces());

  const auto readInData_d = o::Reals(readInData);

  auto fill = OMEGA_H_LAMBDA(o::LO fid) {

    //TODO check if faceids are sequential numbers
    if(!side_is_exposed[fid]) {
      tag_d[fid] = 0;
      return;
    }

    // TODO storing fields at centroid may not be best for long tets.
    auto pos = p::find_face_centroid(fid, coords, face_verts);

    //Cylindrical symmetry = true ? TODO
    o::Real val = p::interpolate2dField(readInData_d, rMin, zMin, dr, dz, nR, nZ, pos, true);
    tag_d[fid] = val; 

    if(verbose > 4 && fid<10){
      printf(" tag_d[%d]= %.5f\n", fid, val);
    }

  };
  o::parallel_for(mesh.nfaces(), fill, "Fill face Tag");

  o::Reals tag(tag_d);
  mesh.set_tag(o::FACE, tagName, tag);
 
}

void GitrmMesh::load1DFieldOnVtxFromFile(const std::string &file, FieldStruct &fs) {
  o::LO verbose = 1;

  std::cout<< "Loading " << fs.name << " from " << file << " on vtx\n" ;
  // Not per vertex; but for input EField: [nZ*nR]
  o::HostWrite<o::Real> readInData;

  processFieldFile(file, readInData, fs, 1);
  std::string tagName = fs.name;
  o::Real rMin = fs.rMin;
  o::Real rMax = fs.rMax;
  o::Real zMin = fs.zMin;
  o::Real zMax = fs.zMax;
  int nR = fs.nR;
  int nZ = fs.nZ;
  OMEGA_H_CHECK(fs.nR >0 && fs.nZ >0);
  o::Real dr = (rMax - rMin)/nR;
  o::Real dz = (zMax - zMin)/nZ;
  if(verbose >2){
    printf(" dr%.5f , dz%.5f , rMax%.5f , rMin%.5f , zMax%.5f, zMin%.5f \n",
        dr, dz, rMax, rMin,zMax , zMin);
  }
  // Interpolate at vertices and Set tag
  o::Write<o::Real> tag_d(mesh.nverts());

  const auto coords = mesh.coords(); //Reals
  const auto readInData_d = o::Reals(readInData);

  auto fill = OMEGA_H_LAMBDA(o::LO iv) {
    o::Vector<3> pos = o::zero_vector<3>();
    // coords' size is 3* nverts
    for(o::LO j=0; j<3; ++j){
      pos[j] = coords[3*iv+j];
      if(verbose > 3 && iv<5){
        printf(" iv:%d %.5f \n", iv, pos[j]);
      }
    }

    //Cylindrical symmetry = true ? TODO
    o::Real val = p::interpolate2dField(readInData_d, rMin, zMin, dr, dz, 
      nR, nZ, pos, true);
    tag_d[iv] = val; 

    if(verbose > 4 && iv<10){
      printf(" tag_d[%d]= %.5f\n", iv, val);
    }
  };
  o::parallel_for(mesh.nverts(), fill, "Fill Tag");

  o::Reals tag(tag_d);
  mesh.set_tag(o::VERT, tagName, tag);
}


void GitrmMesh::addTagAndLoadData(const std::string &profileFile, 
  const std::string &profileFileDensity) {

  mesh.add_tag<o::Real>(o::FACE, "ne", 1); 
  mesh.add_tag<o::Real>(o::FACE, "density", 1); //=ni 
  mesh.add_tag<o::Real>(o::FACE, "Tion", 1);
  mesh.add_tag<o::Real>(o::FACE, "Tel", 1);

  mesh.add_tag<o::Real>(o::VERT, "densityVtx", 1);
  mesh.add_tag<o::Real>(o::VERT, "neVtx", 1);
  mesh.add_tag<o::Real>(o::VERT, "TionVtx", 1);
  mesh.add_tag<o::Real>(o::VERT, "TelVtx", 1);

  FieldStruct fd{"density", "n_x", "n_z", "gridx", "gridz", "ni", "", ""};
  loadScalarFieldOnBdryFaceFromFile(profileFileDensity, fd); 

  FieldStruct fdv{"densityVtx", "n_x", "n_z", "gridx", "gridz", "ni", "", ""};
  load1DFieldOnVtxFromFile(profileFileDensity, fdv);

  FieldStruct fne{"ne", "nR", "nZ", "gridR", "gridZ", "ne", "", ""};
  loadScalarFieldOnBdryFaceFromFile(profileFile, fne); 

  FieldStruct fnev{"neVtx", "nR", "nZ", "gridR", "gridZ", "ne", "", ""};
  load1DFieldOnVtxFromFile(profileFile, fnev);


  // Load ion Temperature
  FieldStruct fti{"Tion", "nR", "nZ", "gridR", "gridZ", "ti", "", ""};
  loadScalarFieldOnBdryFaceFromFile(profileFile, fti); 

  FieldStruct ftiv{"TionVtx", "nR", "nZ", "gridR", "gridZ", "ti", "", ""};
  load1DFieldOnVtxFromFile(profileFile, ftiv);

  // Load electron Temperature
  FieldStruct fte{"Tel", "nR", "nZ", "gridR", "gridZ", "te", "", ""};
  loadScalarFieldOnBdryFaceFromFile(profileFile, fte); 

  FieldStruct ftev{"TelVtx", "nR", "nZ", "gridR", "gridZ", "te", "", ""};
  load1DFieldOnVtxFromFile(profileFile, ftev);

}

//TODO spli this function
void GitrmMesh::initBoundaryFaces() {
  o::LO verbose = 1;

  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto down_r2fs = mesh.ask_down(3, 2).ab2b;

  o::Real bxz[4] = {BGRIDX0, BGRIDZ0, BGRID_DX, BGRID_DZ};
  o::LO bnz[2] = {BGRID_NX, BGRID_NZ};
  const auto &Bfield_2dm = Bfield_2d;

  const auto density = mesh.get_array<o::Real>(o::FACE, "density"); //=ni
  const auto ne = mesh.get_array<o::Real>(o::FACE, "ne");
  const auto te = mesh.get_array<o::Real>(o::FACE, "Tel");
  const auto ti = mesh.get_array<o::Real>(o::FACE, "Tion");

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
      auto pos = p::find_face_centroid(fid, coords, face_verts);

      // TODO angle is between surface normal and magnetic field at center of face
      // If  face is long, BField is not accurate. Calculate at closest point ?
      p::interp2dVector(Bfield_2dm,  bxz[0], bxz[1], bxz[2], bxz[3], bnz[0], bnz[1], 
           pos, B, true);
      if(verbose > 3 ) { //&& fid%1000==0){
        printf(" fid:%d::  %.5f %.5f %.5f tel:%.4f B:%g %g %g\n", fid, pos[0], pos[1], pos[2], te[fid],
            B[0], B[1], B[2]);
      }
      /*
      auto elmId = p::elem_of_bdry_face(fid, f2r_ptr, f2r_elem);
      auto surfNorm = p::find_face_normal(fid, elmId, coords, mesh2verts, 
          face_verts, down_r2fs);
      */
      //TODO verify
      auto surfNorm = p::find_dbry_face_normal(fid,coords,face_verts);
      o::Real magB = o::norm(B);
      o::Real magSurfNorm = o::norm(surfNorm);
      o::Real angleBS = p::osh_dot(B, surfNorm);
      o::Real theta = acos(angleBS/(magB*magSurfNorm));
      if (theta > o::PI * 0.5) {
        theta = abs(theta - o::PI);
      }
      angle_d[fid] = theta*180.0/o::PI;
      
      if(verbose >3) {
        printf("fid:%d surfNorm:%g %g %g angleBS=%g theta=%g angle=%g\n", fid, surfNorm[0], 
          surfNorm[1], surfNorm[2],angleBS,theta,angle_d[fid]);
      }

      o::Real tion = ti[fid];
      o::Real tel = te[fid];
      o::Real nel = ne[fid];
      o::Real dens = density[fid];  //3.0E+19 ?
      o::Real pot = BIAS_POTENTIAL;

      o::Real dlen = 0;
      if(p::almost_equal(nel, 0.0)){
        dlen = 1.0e12;
      }
      else {
        dlen = sqrt(8.854187e-12*tel/(nel*pow(background_Z,2)*1.60217662e-19));
      }
      debyeLength_d[fid] = dlen;

      larmorRadius_d[fid] = 1.44e-4*sqrt(background_amu*tion/2)/(background_Z*magB);
      flux_d[fid] = 0.25* dens *sqrt(8.0*tion*1.60217662e-19/(o::PI*background_amu));
      impacts_d[fid] = 0.0;

      if(BIASED_SURFACE) {
        o::Real cld = 0;
        if(tel > 0.0) {
          cld = dlen * pow(abs(pot)/tel, 0.75);
        }
        else { 
          cld = 1e12;
        }
        childLangmuirDist_d[fid] = cld;
      }
      else {
        pot = 3.0*tel; 
      }
      potential_d[fid] = pot;

      if(verbose >4)// || p::almost_equal(angle_d[fid],0))
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
  mesh.add_tag<o::Real>(o::FACE, "gridE", 1); //TODO store subgrid data ?
  mesh.add_tag<o::Real>(o::FACE, "gridA", 1); //TODO store subgrid data ?
  mesh.add_tag<o::Real>(o::FACE, "sumParticlesStrike", 1);
  mesh.add_tag<o::Real>(o::FACE, "sumWeightStrike", 1);
  mesh.add_tag<o::Real>(o::FACE, "grossDeposition", 1);
  mesh.add_tag<o::Real>(o::FACE, "grossErosion", 1);
  mesh.add_tag<o::Real>(o::FACE, "aveSputtYld", 1);
  mesh.add_tag<o::LO>(o::FACE, "sputtYldCount", 1);

  // TODO get nEdist, nAdist;
  o::LO nEdist = 4; //TODO
  o::LO nAdist = 4; //TODO
  mesh.add_tag<o::Real>(o::FACE, "energyDistribution", nEdist*nAdist);
  mesh.add_tag<o::Real>(o::FACE, "sputtDistribution", nEdist*nAdist);
  mesh.add_tag<o::Real>(o::FACE, "reflDistribution", nEdist*nAdist);
}


// These are only for temporary writing during pre-processing
void GitrmMesh::initNearBdryDistData() {
  OMEGA_H_CHECK(BDRYFACE_SIZE > 0);
  OMEGA_H_CHECK(SIZE_PER_FACE > 0);
  auto nel = mesh.nelems();
  // Flat arrays per element. Copying is passing device data.
  bdryFacesW = o::Write<o::Real>(nel * BDRYFACE_SIZE * (SIZE_PER_FACE-1), 0);
  numBdryFaceIds = o::Write<o::LO>(nel, 0);
  // This size is also pre-decided
  bdryFaceIds = o::Write<o::LO>(nel * BDRYFACE_SIZE, -1);
  bdryFlags = o::Write<o::LO>(nel, 0);
  bdryFaceElemIds = o::Write<o::LO>(nel * BDRYFACE_SIZE, -1);
}



// No more changes to the bdry faces data; and delete flat arrays
void GitrmMesh::convert2ReadOnlyCSR() {
  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);

  //const auto &bdryFacesW = this->bdryFacesW;  //not used now
  const auto &numBdryFaceIds = this->numBdryFaceIds;
  const auto &bdryFaceIds = this->bdryFaceIds;
  const auto &bdryFaceElemIds = this->bdryFaceElemIds;

  auto &bdryFaces = this->bdryFaces;
  auto &bdryFaceInds = this->bdryFaceInds;
  o::LO S = SIZE_PER_FACE;
  // Convert to CSR.
  // Not actual index of bdryFaces, but has to x by SIZE_PER_FACE to get it
  o::LOs numBdryFaceIds_r(numBdryFaceIds);
  numAddedBdryFaces = calculateCsrIndices(numBdryFaceIds_r, bdryFaceInds);

  //Keep separate copy of bdry face coords per elem, to avoid  accessing another
  // element's face. Can save storage if common bdry face data is used by face id?
  // Replace flat arrays of Write<> by convenient,
  // Kokkos::View<double[numAddedBdryFaces][SIZE_PER_FACE]> ..  ?

  o::Write<o::Real> bdryFacesCsrW(numAddedBdryFaces * S, 0);
  
  //Copy data
  auto convert = OMEGA_H_LAMBDA(o::LO elem) {
    // Index points to data counting spread out face or x,y,z components. ?
    // This calculates face number from the above. 

    // face index is NOT the actual array index of face data.
    auto beg = bdryFaceInds[elem];
    auto end = bdryFaceInds[elem+1];
    if(beg == end){
      return;
    }
    OMEGA_H_CHECK((end - beg) == numBdryFaceIds[elem]);

    // Source position calculated using pre-assigned fixed number of faces
    // per element. Here face number (outer loop) is used to get absolute position.
    // Size_per_face was effectively 2 less so far, since the face id was not added.
    // The face_id is added as Real at the 1st entry of each face.

    // Source face size is 1 less.
    //auto bfd = elem*BDRYFACE_SIZE*(S-1);
    auto bfi = elem*BDRYFACE_SIZE;
    o::Real fdat[9] = {0};

    // Looping over implicit face numbers
    for(o::LO i = beg; i < end; ++i) {
      p::get_face_data_by_id(face_verts, coords, bdryFaceIds[bfi], fdat, 0);

      // Face index is x by size_per_face 
      bdryFacesCsrW[i*S] = static_cast<o::Real>(bdryFaceIds[bfi]);
      bdryFacesCsrW[i*S+1] = static_cast<o::Real>(bdryFaceElemIds[bfi]);
      //printf(" f_%d  el_%d\n", bdryFaceIds[bfi], bdryFaceElemIds[bfi]);
      for(o::LO j = FSKIP; j < S; ++j) {
        //bdryFacesCsrW[i*S + j] = bdryFacesW[bfd + j-1]; //TODO check this
        bdryFacesCsrW[i*S + j] = fdat[j - FSKIP];
      }
      ++bfi;
    }    
  };
  o::parallel_for(mesh.nelems(), convert, "Convert to CSR write");
  //CSR data. Conversion is making RO object on device by passing device data.
  bdryFaces = o::Reals(bdryFacesCsrW);

}



// Pre-processing to fill bdry faces in elements near bdry within depth needed. 
// Call init function before this. Filling flat arrays; Later call convert2ReadOnlyCSR.

void GitrmMesh::copyBdryFacesToSelf() {

  MESHDATA(mesh);
  //auto &bdryFacesW = this->bdryFacesW;
  auto &numBdryFaceIds = this->numBdryFaceIds;
  auto &bdryFaceIds = this->bdryFaceIds;
  auto &bdryFaceElemIds = this->bdryFaceElemIds;
  auto &bdryFlags = this->bdryFlags;

  o::LO verbose = 1;
  auto fillBdry = OMEGA_H_LAMBDA(o::LO elem) {

    // Find  bdry faces: # and id
    o::LO nbdry = 0; 
    o::LO bdryFids[4];
    const auto beg_face = elem *4;
    const auto end_face = beg_face +4;
    for(o::LO fi = beg_face; fi < end_face; ++fi) {//not 0..3
      const auto fid = down_r2f[fi];
      if(verbose>2) {
        printf( "exposed: e_%d, i_%d, fid_%d => %d\n", elem, fi, fid, side_is_exposed[fid]);
      }
      if(side_is_exposed[fid]) {
        bdryFids[nbdry] = fid;
        ++nbdry;
      }
    }

    // If no bdry faces to add
    if(!nbdry){
      return;
    }
    if(verbose>2) printf("\nelem:%d, nbdry: %d \n", elem, nbdry);

    //Only boundary faces
    for(o::LO fi = 0; fi < nbdry; ++fi) {
      auto fid = bdryFids[fi];
      auto fv2v = o::gather_verts<3>(face_verts, fid); //Few<LO, 3>

      // const auto face = o::gather_vectors<3, 3>(coords, fv2v);
      //From self bdry

      // Function call forces the data passed to be const
      // p::addFaceToBdryData(bdryFacesW, bdryFaceIds, BDRYFACE_SIZE, SIZE_PER_FACE, 
     //  3, fi, fid, elem, face);
      /*
      //TODO test this
      for(o::LO i=0; i<3; ++i){
        for(o::LO j=0; j<3; j++){
            bdryFacesW[elem* BDRYFACE_SIZE* (SIZE_PER_FACE-1)  + 
              fi* (SIZE_PER_FACE-1) + i*3 + j] = face[i][j];
            if(verbose>2)
                printf("elem_%d FACE %d: %d %d %0.3f  @%d \n" , elem, fid ,i,j, face[i][j],
                elem* BDRYFACE_SIZE* (SIZE_PER_FACE-1)  +  fi* (SIZE_PER_FACE-1) + i*3 + j); 
        }
      }
      
      for(int fv=0; fv< 3;++fv){ 
        for(int vc=0; vc< 3;++vc){
          printf( "bdryFacesW: e_%d, fid_%d => %0.4f\n", elem, fid, 
            bdryFacesW[elem* BDRYFACE_SIZE* (SIZE_PER_FACE-1)  + 
                fi* (SIZE_PER_FACE-1) + fv*3 + vc]);}}
     
      */
      auto ind = elem* BDRYFACE_SIZE + fi;
      bdryFaceIds[ind] = fid;
      bdryFaceElemIds[ind] = elem;

    }
    numBdryFaceIds[elem] = nbdry;
    if(verbose>2)
      printf( "numBdryFaceIds: e_%d, num_%d \n", elem, numBdryFaceIds[elem]);

    //Set neigbhor's flags
   // p::updateAdjElemFlags(dual_elems, dual_faces, elem, bdryFlags, nbdry);
    auto dface_ind = dual_elems[elem];
    for(o::LO i=0; i<4-nbdry; ++i){
      auto adj_elem  = dual_faces[dface_ind];
      o::LO val = 1;
      Kokkos::atomic_exchange( &bdryFlags[adj_elem], val);
      if(verbose>2)
        printf( "bdryFlags: e_%d : %d ; set_by_%d\n", adj_elem, bdryFlags[adj_elem], elem);
      ++dface_ind;
    }

  };
  o::parallel_for(nel, fillBdry, "CopyBdryFacesToSelf");
  
}


void GitrmMesh::preProcessDistToBdry() {
  MESHDATA(mesh);
  //auto &bdryFacesW = this->bdryFacesW;
  auto &numBdryFaceIds = this->numBdryFaceIds;
  auto &bdryFaceIds = this->bdryFaceIds;
  auto &bdryFaceElemIds = this->bdryFaceElemIds;
  auto &bdryFlags = this->bdryFlags;
  
  initNearBdryDistData();
  copyBdryFacesToSelf();

  auto fill = OMEGA_H_LAMBDA(o::LO elem) {
    o::LO verbose = 0;
    // This has to be init by copyBdryFacesToSameElem.
    o::LO update = bdryFlags[elem];
    if(verbose >2) 
      printf( "\n\n Beg:bdryFlags: e_%d, update: %d nbf_%d\n", elem, 
        bdryFlags[elem], numBdryFaceIds[elem]);

    //None of neigbhors have update
    if(! update){
      return;
    }
    o::LO val = 0;
    Kokkos::atomic_exchange(&bdryFlags[elem], val);

    const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, elem);
    const auto tet = Omega_h::gather_vectors<4, 3>(coords, tetv2v);

    o::LO nbf = 0;
    
    // Bdry element need not have regions that have shortest dist to bdry
    // always on the own bdry face. So, need to add adj. faces. So, interior
    // faces of bdry elems have to add neigbhors' faces

    // Store existing # of faces
    o::LO existing = numBdryFaceIds[elem];
    o::LO tot = existing;

    o::LO inFids[4] = {-1, -1, -1, -1};
    // One purpose of small functions that use mesh data is to pre-use 
    // mesh data as far as possible, i.e. minimize using mesh data 
    // alongside unavoidable large-data in a loop.
    o::LO nIn = p::get_face_type_ids_of_elem(elem, down_r2f, 
                  side_is_exposed, inFids, INTERIOR);
    nbf = 4 - nIn;

    auto dface_ind = dual_elems[elem]; //first interior

    for(auto fi = 0; fi < nIn; ++fi) {

      //Check all adjacent elements
      auto adj_elem  = dual_faces[dface_ind];

      //auto fid = inFids[fi];
      //auto fv2v = o::gather_verts<3>(face_verts, fid); //Few<LO, 3>
      //const auto face = o::gather_vectors<3, 3>(coords, fv2v);

      // Get ids updated by adj. elems of previous faces processed
      o::LO sizeAdj = numBdryFaceIds[adj_elem]; 

      if(verbose >2) 
        printf("\n **tot:%d , SizeAdj_of_%d = %d \n", tot, adj_elem, sizeAdj);

      for(o::LO ai=0; ai < sizeAdj; ++ai) {
        bool found = false;
        auto index = BDRYFACE_SIZE*adj_elem + ai;
        auto adj_fid = bdryFaceIds[index];
        auto adj_fElid = bdryFaceElemIds[index];

         if(verbose >2) 
          printf("adj_elem_%d, fadj_ %d, of_e_%d : ", adj_elem,  adj_fid, adj_fElid); 

        for(auto ii = 0; ii < tot; ++ii){
          if(verbose >2) printf("   this_fid_ %d ", bdryFaceIds[BDRYFACE_SIZE*elem+ii]);
          if(adj_fid == bdryFaceIds[BDRYFACE_SIZE*elem+ii]){ 
            if(verbose >2) printf("found %d in current \n", adj_fid);
            found = true;
            break;
          }
        }
        if(found) { 
          continue;
        }
        if(verbose >2) printf("NOT found \n");

        // In current element
        //TODO test this
        //o::LO start = adj_elem*BDRYFACE_SIZE*(SIZE_PER_FACE-1) + ai*(SIZE_PER_FACE-1);
        //if(verbose >2) printf("start@%d elem_%d adj_elem_%d ai_%d\n", start, elem, adj_elem, ai);
        bool add = p::check_if_face_within_dist_to_tet(tet, face_verts, coords, 
                    adj_fid, DEPTH_DIST2_BDRY);
        if(!add) {
          if(verbose >2) printf("NOT adding \n");
          continue;
        }       
        if(verbose >2) 
          printf("From_%d ADDING.. fid %d of_e_%d\n", adj_elem , adj_fid, adj_fElid);
        /*
        // This step is not needed. Currently this data is not used
        auto fv2v = o::gather_verts<3>(face_verts, adj_fid); //Few<LO, 3>
        const auto face = o::gather_vectors<3, 3>(coords, fv2v);
        for(o::LO i=0; i<3; ++i){
          for(o::LO j=0; j<3; j++){
            bdryFacesW[elem* BDRYFACE_SIZE*(SIZE_PER_FACE-1) 
              + tot* (SIZE_PER_FACE-1) + i*3 + j] = face[i][j];
          }
        }
        */
        auto ind = elem* BDRYFACE_SIZE + tot;
        bdryFaceIds[ind] = adj_fid;
        bdryFaceElemIds[ind] = adj_fElid;
        ++tot;

        if(verbose >2) { 
          if(numBdryFaceIds[elem]>0)
            printf(" %d:: %d;ids ", elem, numBdryFaceIds[elem]);
          printf("tot = %d\n", tot);
        }

      }
      ++dface_ind;
    }
    if(tot > existing){ 
      if(verbose >2)  printf("numBdryFaceIds_of_%d = %d\n", elem, tot);
      numBdryFaceIds[elem] =  tot;

      // After copying is done
      auto dface_ind = dual_elems[elem];
      for(o::LO i=0; i<4-nbf; ++i) {
        auto adj_elem  = dual_faces[dface_ind];
        o::LO val = 1;
        Kokkos::atomic_exchange( &bdryFlags[adj_elem], val);
        ++dface_ind;
      }
    }
    if(verbose >2)
      if(numBdryFaceIds[elem]>0)printf("> %d:: %d;ids \n\n", elem, numBdryFaceIds[elem]);

  };

  o::LO total = 1;
  o::LO ln = 0;
  while(total){
    o::parallel_for(nel, fill, "FillBdryFacesInADepth"); //4,690

   
  // Return type error
  //  total = o::parallel_reduce(nel, OMEGA_H_LAMBDA(
  //    const int& i, o::LO& update){
  //      update += bdryFlags[i];  //()
  //  }, "BdryFillCheck");
  
   
   Kokkos::parallel_reduce("BdryFillCheck", nel, 
       OMEGA_H_LAMBDA(const int& i, o::LO & lsum) {
     lsum += bdryFlags[i];
   }, total);

   printf("LOOP %d total %d \n", ln, total);
   ++ln;
  }
  
  convert2ReadOnlyCSR();
}


void GitrmMesh::printBdryFaceIds(bool printIds, o::LO minNums) {
  auto &numBdryFaceIds = this->numBdryFaceIds;
  auto &bdryFaceIds = this->bdryFaceIds;
  auto &bdryFaceElemIds = this->bdryFaceElemIds;
  auto print = OMEGA_H_LAMBDA(o::LO elem){
    o::LO num = numBdryFaceIds[elem];
    if(num > minNums){
      printf(" %d (tot %d ) :: ", elem, numBdryFaceIds[elem]);
      if(printIds){
        for(o::LO i=0; i<num; ++i){
          o::LO ind = elem * BDRYFACE_SIZE + i;
          printf("%d( %d ), ", bdryFaceIds[ind], bdryFaceElemIds[ind]);
        }
      }
      printf("\n");
    }
  };
  
  o::parallel_for(mesh.nelems(), print, "printBdryFaceIds");
  printf("\nDEPTH: %0.5f \n", DEPTH_DIST2_BDRY);
}


void GitrmMesh::printBdryFacesCSR(bool printIds, o::LO minNums) {
  auto &bdryFaceIds = this->bdryFaceIds;
  auto &bdryFaceElemIds = this->bdryFaceElemIds;
  auto &bdryFaceInds = this->bdryFaceInds;
  auto print = OMEGA_H_LAMBDA(o::LO elem){
    o::LO ind1 = bdryFaceInds[elem];
    o::LO ind2 = bdryFaceInds[elem+1];
    auto nf = ind2 - ind1;

    if(nf > minNums){
      printf("e_%d #%d  %d  %d \n", elem, nf, ind1, ind2);
      if(printIds){
        for(o::LO i=0; i<nf; ++i){
          o::LO ind = elem * BDRYFACE_SIZE+i;
          printf("%d( %d ), ", bdryFaceIds[ind], bdryFaceElemIds[ind]);
        }
      }
      printf("\n");
    }
  };
  printf("\nCSR: \n");
  o::parallel_for(mesh.nelems(), print, "printBdryFacesCSR");
  printf("\nDEPTH: %0.5f \n", DEPTH_DIST2_BDRY);
}





/*

// Convert indices to CSR, face ids are not yet available
// TODO this method overlaps with part of convertToCsr()
void GitrmMesh::preProcessCreateIndexDistToBdry(){
  auto &numBdryFaceIdsAdjs = this->numBdryFaceIdsAdjs;
  auto &totalBdryFacesCsrAdjs = this->totalBdryFacesCsrAdjs;
  auto &bdryFacesCsrW  = this->bdryFacesCsrW;
  auto &bdryFaceIndsAdjs = this->bdryFaceIndsAdjs;

  OMEGA_H_CHECK(countedAdjs);

  //Copy to host
  o::HostRead<o::LO> numBdryFaceIdsAdjsH(numBdryFaceIdsAdjs);
  o::HostWrite<o::LO> bdryFaceIndsAdjsH(mesh.nelems()+1);
  //Num of faces
  o::Int sum = 0;
  o::Int e = 0;
  for(e=0; e < mesh.nelems(); ++e){
    bdryFaceIndsAdjsH[e] = sum; // * S;
    sum += numBdryFaceIdsAdjsH[e];
  }
  // last entry
  bdryFaceIndsAdjsH[e] = sum;

  totalBdryFacesCsrAdjs = sum;

  //CSR indices
  bdryFaceIndsAdjs = o::LOs(bdryFaceIndsAdjsH.write());

  bdryFacesCsrW = o::Write<o::Real>(totalBdryFacesCsrAdjs * SIZE_PER_FACE, 0);
}


// TODO this method overlaps with part of convertToCsr()
void GitrmMesh::preProcessFillDataDistToBdry(){
  MESHDATA(mesh);
  
  auto &numBdryFaceIdsAdjs = this->numBdryFaceIdsAdjs;
  auto &bdryFacesAdjs = this->bdryFacesAdjs;
  auto &bdryFaceIndsAdjs = this->bdryFaceIndsAdjs;
  auto &bdryFacesCsrW = this->bdryFacesCsrW;

  auto S = SIZE_PER_FACE;

  //Copy data
  auto convert = OMEGA_H_LAMBDA(o::LO elem){
    auto beg = bdryFaceIndsAdjs[elem];
    auto end = bdryFaceIndsAdjs[elem+1];
    if(beg == end){
      return;
    }
    OMEGA_H_CHECK((end - beg) == numBdryFaceIdsAdjs[elem]);
    o::Real fdat[9] = {0};
    // Looping over implicit face numbers
    for(o::LO ind = beg; ind < end; ++ind){
      auto faceId = static_cast<o::LO>(bdryFacesCsrW[ind*S]);
      p::get_face_data_by_id(face_verts, coords, faceId, fdat, 0);
      for(o::LO j = FSKIP; j < S; ++j){
        //bdryFacesCsrW[i*S + j] = bdryFacesW[bfd + j-1]; //TODO check this
        bdryFacesCsrW[ind*S + j] = fdat[j - FSKIP];
      }
    }    
  };
  o::parallel_for(nel, convert, "Convert to CSR write");
  bdryFacesAdjs = o::Reals(bdryFacesCsrW);

}

*/
/** @brief Find distance to boundary by a redundant method
*   Current boundary face's element is taken, and its adjacent elements
*   are found by region -> face -> region
*/
/*
void GitrmMesh::preProcessSearchDistToBdryAdjs(o::LO step){
  MESHDATA(mesh);

  auto &numBdryFaceIdsAdjs = this->numBdryFaceIdsAdjs;
  auto &bdryFaceIndsAdjs = this->bdryFaceIndsAdjs;
  auto &bdryFacesCsrW = this->bdryFacesCsrW;

  o::Write<o::LO>indexWritten;

  auto S = SIZE_PER_FACE;

  // Count only
  if(step==1){
    // initializing to 0 is important
    numBdryFaceIdsAdjs = o::Write<o::LO>(mesh.nelems(), 0);
  }
  // Fill only
  else if(step==2){
    // initializing to 0 is important
    indexWritten = o::Write<o::LO>(mesh.nelems(), 0);

    preProcessCreateIndexDistToBdry();
  }


  auto run = OMEGA_H_LAMBDA(o::LO elem){
    // Collect bdry faces
    o::LO bdryFids[4];
    o::LO nbdry = p::get_face_type_ids_of_elem(elem, down_r2f, side_is_exposed, bdryFids, EXPOSED);
    if(nbdry ==0)
      return;

    o::LO nIn = 0;
    // Current boundary face of current element
    for(auto bfi=0; bfi < nbdry; ++bfi){

      o::LO bfsElems[BFS_DATA_SIZE];
      auto bfid = bdryFids[bfi];

      o::LO current = -1;
      o::LO nAdded = 0;

      // Current element is added to BFS. Repeated per bfid
      bfsElems[nAdded++] = elem;

      //Current bdry face numbers (=1) is added to this elem
      if(step==1){
        Kokkos::atomic_add(&numBdryFaceIdsAdjs[elem], 1);
      }

      // Breadth First Search
      while(true){
        current += 1;
        if(current >= nAdded)
          break;

        // Implicit pop
        o::LO e = bfsElems[current];

        // Number of interior faces and its adj. element ids
        nIn = p::get_face_type_ids_of_elem(e, down_r2f, side_is_exposed, bdryFids, INTERIOR);

        // First interior. All are interior faces, CSR entry is certain 
        auto dface_ind = dual_elems[e]; 

        for(auto fi=0; fi < nIn; ++fi){

          // Adjacent element across this face fi
          auto adj_elem  = dual_faces[dface_ind];
          ++dface_ind;

          // Adj. element tet
          const auto tetv2v = Omega_h::gather_verts<4>(mesh2verts, adj_elem);
          const auto tet = Omega_h::gather_vectors<4, 3>(coords, tetv2v);

          // Any vertex of Tet within depth limit  
          bool add = p::check_if_face_within_dist_to_tet(tet, face_verts, coords, 
                      bfid, DEPTH_DIST2_BDRY);
          if(!add) 
            continue;
          // Add the reference face to this adj_elem.

          if(step==1){
            Kokkos::atomic_add(&numBdryFaceIdsAdjs[adj_elem],  1);
          }
          else if(step==2){
            auto nf = numBdryFaceIdsAdjs[e];

            // Current position for writing face ids in this element. This data is not updated in
            // the current call of step=2
            auto beg = bdryFaceIndsAdjs[e];
            auto next = bdryFaceIndsAdjs[e+1];

            //Keep track of index written, v is old value.
            //TODO error o::atomic_fetch_add()
            int v = Kokkos::atomic_fetch_add(&indexWritten[e], 1);            
            // Method to guarantee reserving data for use by this thread : 
            // https://devtalk.nvidia.com/default/topic/850890
            //     /can-one-force-two-operations-to-occur-atomically-together-/
            if(v < nf){
              // These steps OK between atomic operations ?
              auto pos = beg + v; 
              OMEGA_H_CHECK(pos < next);
              //TODO omega_h has no file=exchange ?
              Kokkos::atomic_exchange(&bdryFacesCsrW[S*pos], static_cast<o::Real>(bfid));
              //Element id that of the bfid
              Kokkos::atomic_exchange(&bdryFacesCsrW[S*pos +1], static_cast<o::Real>(elem));

            }
          }

          // Add this  adj_elem to bfs
          bfsElems[nAdded++] = adj_elem;
        }
      }
    }
  };
  o::parallel_for(mesh.nelems(), run, "preProcessDistToBdryAdjSearch");
  if(step == 1)
    countedAdjs = true;
}


void GitrmMesh::preProcessDistToBdryAdjs(){

    preProcessSearchDistToBdryAdjs(1);
    preProcessSearchDistToBdryAdjs(2);
    preProcessFillDataDistToBdry();

}
//Add in hpp
  // Adjacency based BFS search, for testing the pre-processing
  o::Reals bdryFacesAdjs;
  o::LOs bdryFaceIndsAdjs;

  void preProcessDistToBdryAdjs();
  void preProcessSearchDistToBdryAdjs(o::LO);
  void preProcessFillDataDistToBdry();
  void preProcessCreateIndexDistToBdry();
  o::Write<o::LO> numBdryFaceIdsAdjs;
  o::Write<o::Real> bdryFacesCsrW;
  bool countedAdjs = false;
  int totalBdryFacesCsrAdjs = 0;

*/