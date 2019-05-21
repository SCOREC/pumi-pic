#include <fstream>
#include <algorithm>

#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "Omega_h_reduce.hpp"
//#include "Omega_h_atomics.hpp"  //No such file
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Kokkos_Core.hpp>

namespace o = Omega_h;
namespace p = pumipic;


//o::Mesh* GitrmMesh::mesh = nullptr;
//bool GitrmMesh::hasMesh = false;

GitrmMesh::GitrmMesh(o::Mesh &m): 
  mesh(m) {
}


void GitrmMesh::loadFieldsNBoundary() {

}


// BField is to be stored on vertices in 3D; source is 2D
void GitrmMesh::processBFieldFile(const std::string &bFile, 
    o::HostWrite<o::Real> &data, o::Real &rMin, o::Real &rMax, 
    o::Real &zMin,  o::Real &zMax, int &nR, int &nZ) {
  std::ifstream ifs(bFile);
  if (!ifs.is_open()) {
    std::cout << "Error opening BField file " << bFile << '\n';
    //exit(1);
  }
  o::LO verbose = 0;

  int n = 0;
  bool br, bt, bz;
  br = bt = bz = false;
  o::LO ind = 0;
  std::string sd = "";

  std::string line;
  std::string s1, s2, s3;
  while(std::getline(ifs, line)) {
    if(verbose >3)
      std::cout << "Processing  line " << line << '\n';
    std::replace (line.begin(), line.end(), ',' , ' ');
    std::replace (line.begin(), line.end(), ';' , ' ');

    std::stringstream ss(line);
    ss >> s1;
    if(verbose >3)
      std::cout << "str s1:" << s1 << "\n";
   
    if(s1.find_first_not_of(' ') == std::string::npos ){
      s1 = "";
      continue;
    }

    if(s1 == "nR") {
      ss >> s2 >> s3;
      nR = std::stoi(s3);
    }
    else if(s1 == "nZ") {
      ss >> s2 >> s3;
      nZ = std::stoi(s3);
    }
    else if(s1 == "rMin") {
      ss >> s2 >> s3;
      rMin = std::stod(s3);
    }    
    else if(s1 == "rMax") {
      ss >> s2 >> s3;
      rMax = std::stod(s3);
    } 
    else if(s1 == "zMin") {
      ss >> s2 >> s3;
      zMin = std::stod(s3);
    } 
    else if(s1 == "zMax") {
      ss >> s2 >> s3;
      zMax = std::stod(s3);
    }
    else if(s1 == "br") {
      // br,bt,bz(nZ, nR)
      // If "br" is not seen before other components, it is error
      ss >> s2;
      data = o::HostWrite<o::Real>(3*nR*nZ);
      ind = 0;
      while(ss >> sd){
        data[3*ind] = std::stod(sd);
        ++ind;
      }
      br = true;
    }
    else if(br) {
      if(s1 == "bt") {
        br = false;
        bt = true;
      }
      else {
        while(ss >> sd){
          data[3*ind] = std::stod(sd);
          ++ind;
        }
      }
    }

    if(s1 == "bt") {
      ind = 0;
      ss >> s2;
      while(ss >> sd){
        data[3*ind+1] = std::stod(sd);
        ++ind;
      }
      bt = true;
    }
    else if(bt) {
      if(s1 == "bz") {
        bt = false;
        bz = true;
      }
      else { 
        while(ss >> sd) {
          data[3*ind+1] = std::stod(sd);
          ++ind;
        }
      }
    }

    if(s1 == "bz") {
      ind = 0;
      ss >> s2;
      while(ss >> sd){
        data[3*ind+2] = std::stod(sd);
        ++ind;
      }
      bz = true;
    }
    else if(bz) {
      if(s1 == "Density_m3") {
        bz = false;
      }
      else {
        while(ss >> sd) {
          data[3*ind+2] = std::stod(sd);
          ++ind;
        }
      }
    }

    s1 = s2 = s3 = "";
    sd = "";
  }

  if(verbose >2){
    std::cout << "\n\n " << nR << " " << nZ << " " << rMin<< " "<< rMax 
              << " "<< zMin << " "<< zMax << "\n\n";
  }

}

void GitrmMesh::loadBField(o::Mesh &mesh, const std::string &bFile) {
  int verbose = 4;
  std::cout<< "Loading BField.. \n" ;
  // Not per vertex; but for input BField:  br,bt,bz[nZ*nR]
  o::HostWrite<o::Real> readInData;
  o::Real rMin = 0; 
  o::Real zMin = 0; 
  o::Real rMax = 0; 
  o::Real zMax = 0; 
  o::LO nR = 0;
  o::LO nZ = 0;
  processBFieldFile(bFile, readInData, rMin, rMax, zMin, zMax, nR, nZ);
  OMEGA_H_CHECK(nR >0 && nZ >0);
  o::Real dr = (rMax - rMin)/nR;
  o::Real dz = (zMax - zMin)/nZ;

  // bFields interpolate at vertices and Set tag
  //auto tagB =  o::Write<o::Real>(mesh.nverts()*3, 0);
  o::HostWrite<o::Real> tagB(3*mesh.nverts());
  o::Vector<3> fv{0,0,0};
  o::Vector<3> pos{0,0,0};
  const auto coords = mesh.coords(); //Reals
  // coords' size is 3* nverts
  for(int iv=0; iv < mesh.nverts(); ++iv){
    for(int j=0; j<3; ++j){
      pos[j] = coords[3*iv+j];

      if(verbose > 3 && iv<5){
        std::cout<< " iv:" << iv << " " << pos[j]<< " \n";
      }
    }

    if(verbose > 3 && iv<2){
      std::cout<< " :" << rMin << ":" <<zMin<<  ":" <<dr<<  ":" <<
        dz<<  ":" <<nR<<  ":" <<nZ << "\n";
    }

    //Cylindrical symmetry = true
    p::interp2dVectorHost(readInData, rMin, zMin, dr, dz, nR, nZ, pos, fv, true);
    for(int j=0; j<3; ++j){ //components
      tagB[3*iv+j] = fv[j]; 

      if(verbose > 3 && iv<10){
        std::cout<<" tagB[" << 3*iv+j << "]=" << tagB[3*iv+j]<< " \n";
      }
    }
  }

  mesh.set_tag(o::VERT, "BField", o::Reals(tagB));
}


// Tags are not removed anywhere
void GitrmMesh::initFieldsNBoundary(const std::string &bFile) {
  auto nv = mesh.nverts();
  mesh.add_tag<o::Real>(o::VERT, "BField", 3);
  auto nf = mesh.nfaces();
  mesh.add_tag<o::Real>(o::FACE, "angleBdryBfield", 1);
  mesh.add_tag<o::Real>(o::FACE, "potential", 1);
  mesh.add_tag<o::Real>(o::FACE, "DebyeLength", 1);
  mesh.add_tag<o::Real>(o::FACE, "LarmorRadius", 1);
  mesh.add_tag<o::Real>(o::FACE, "ChildLangmuirDist", 1);

  // Load fields
  loadBField(mesh, bFile);


  //TODO temp
  auto temp =  o::Write<o::Real>(mesh.nfaces(), 0);
  mesh.set_tag(o::FACE, "angleBdryBfield", o::Reals(temp));
  mesh.set_tag(o::FACE, "potential", o::Reals(temp));
  mesh.set_tag(o::FACE, "DebyeLength",  o::Reals(temp));
  mesh.set_tag(o::FACE, "LarmorRadius", o::Reals(temp));
  mesh.set_tag(o::FACE, "ChildLangmuirDist",  o::Reals(temp));

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


void GitrmMesh::calculateCsrIndices(const o::Write<o::LO> &numBdryFaceIds, const bool counted,
    o::LOs &bdryFaceInds, o::LO &totalBdryFaces) {
  OMEGA_H_CHECK(counted);

  // This is done on host, since ordered sum of all previous element ids required for CSR.
  // Not actual index of bdryFaces, but has to x by SIZE_PER_FACE to get it
  //Copy to host
  o::HostRead<o::LO> numBdryFaceIdsH(numBdryFaceIds);
  o::HostWrite<o::LO> bdryFaceIndsH(mesh.nelems()+1);
  //Num of faces
  o::Int sum = 0;
  o::Int e = 0;
  for(e=0; e < mesh.nelems(); ++e){
    bdryFaceIndsH[e] = sum; // * S;
    sum += numBdryFaceIdsH[e];
  }
  // last entry
  bdryFaceIndsH[e] = sum;

  totalBdryFaces = sum;

  //CSR indices
  bdryFaceInds = o::LOs(bdryFaceIndsH.write());
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

  auto nel = mesh.nelems();
  o::LO S = SIZE_PER_FACE;

  // Convert to CSR.
  //Copy to host
  o::HostRead<o::LO> numBdryFaceIdsH(numBdryFaceIds);
  // This is done on host, since ordered sum of all previous element ids required for CSR.
  // Not actual index of bdryFaces, but has to x by SIZE_PER_FACE to get it
  o::HostWrite<o::LO> bdryFaceIndsCsr(nel+1);
  //Num of faces
  o::Int sum = 0;
  o::Int e = 0;
  for(e=0; e < nel; ++e) {
    bdryFaceIndsCsr[e] = sum; // * S;
     sum += numBdryFaceIdsH[e];
  }
  // last entry
  bdryFaceIndsCsr[e] = sum;

  numAddedBdryFaces = sum;

  //CSR indices
  bdryFaceInds = o::LOs(bdryFaceIndsCsr.write());

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
  o::parallel_for(nel, convert, "Convert to CSR write");
  //CSR data. Conversion is making RO object on device by passing device data.
  // No fence needed, since not copying to host.
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

  o::LO verbose = 2;
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
       KOKKOS_LAMBDA(const int& i, o::LO & lsum) {
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

