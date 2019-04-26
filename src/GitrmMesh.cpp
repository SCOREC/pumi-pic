#include "GitrmMesh.hpp"
#include "Omega_h_reduce.hpp"


namespace o = Omega_h;
namespace p = pumipic;

/*
bool gitrm_findDistanceToBdry(const gitrmMesh &gm, 
  SellCSigma<Particle>* scs) {

  MESHDATA(gm->mesh);

  const auto bdryFacesData = gm.bdryFaces;
  const auto bdryFacesPtr = gm.bdryFaceInds;
  
  auto *scs = pcl->scs;
  auto &thread = pcl->thread;

  // elem will be declared; pass scs,thread. 
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, elem, {
    o::LO nFaces = 0;
    const auto *bdryFacesOfElem = getBdryFacesOfElem(bdryFaces,
                          bdryFaceInds, elem, nFaces);

    // pid will be declared
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      const auto ref = scs.get(pid); // FIXME 
      oVector point{0, 0, 0};
      o::Few<o::Real, nFaces> dists({0});
      o::Few< oVector, 3> face;

      for( fi = 0; fi < nFaces; ++fi ){
        getTetFaceVectors(bdryFacesOfElem, fi, face);
        //auto v =
        findNearestPointOnTriangle(face, ref, point); 
        //if(v==TriRegion::VERTB) std::cout << "vertex B \n";
        // std::cout << static_cast<int>(v) << "\n";

        dists[face] = osh_dot(point - ref, point - ref);
      }
      o::Real dist = std::min(dists); //FIXME 
      scs.distToBdry[pid] = dist;
    });
  });
}
*/

GitrmMesh::GitrmMesh(o::Mesh &m): 
  mesh(m),
  numNearBdryElems(0),
  numAddedBdryFaces(0) {
    initNearBdryDistData();
}

// These are only for temporary writing during pre-processing
void GitrmMesh::initNearBdryDistData(){
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
  const auto &bdryFacesW = this->bdryFacesW;  //not used ?
  const auto &numBdryFaceIds = this->numBdryFaceIds;
  const auto &bdryFaceIds = this->bdryFaceIds;
  auto &bdryFaces = this->bdryFaces;
  auto &bdryFaceInds = this->bdryFaceInds;

  auto nel = mesh.nelems();
  /*
  // omega_h parallel_reduce not working !
  numAddedBdryFaces = o::parallel_reduce(nel, OMEGA_H_LAMBDA(
    const int i, double& update){
      update += numBdryFaceIds[i];
  }, "numAddedBdryFaces");
  numNearBdryElems = o::parallel_reduce(nel, OMEGA_H_LAMBDA(
    const int i, double& update){
      if(numBdryFaceIds[i] >0){
        update += 1;
      }
  }, "numNearBdryElems");
 */

  Kokkos::parallel_reduce("numAddedBdryFaces", nel, 
       KOKKOS_LAMBDA(const int &i, o::LO &lsum ) {
     lsum += numBdryFaceIds[i];
  }, numAddedBdryFaces);

  Kokkos::parallel_reduce("numNearBdryElems", nel, 
       KOKKOS_LAMBDA(const int &i, o::LO &lsum ) {
     if(numBdryFaceIds[i] >0){
       lsum += 1;
     }
  }, numNearBdryElems);


  OMEGA_H_CHECK(numNearBdryElems > 0);
  OMEGA_H_CHECK(numAddedBdryFaces > 0);

  // Convert to CSR.
  //Copy to host
  o::HostRead<o::LO> numBdryFaceIdsH(numBdryFaceIds);
  // Done on host, since ordered sum of all previous element ids required for CSR.
  // This is per near bdry element
  o::HostWrite<o::LO> bdryFaceIndsCsr(nel+1, 0);
  o::Int sum = 0;
  o::Int e = 0;
  for(e=0; e<nel; ++e){
    bdryFaceIndsCsr[e] = sum * SIZE_PER_FACE;
     sum += numBdryFaceIdsH[e];
  }
  // last entry
  bdryFaceIndsCsr[e] = bdryFaceIndsCsr[e-1];

  //CSR indices
  bdryFaceInds = o::LOs(bdryFaceIndsCsr.write()); //TODO check

  //Keep separate copy of bdry face coords per elem, to avoid  accessing another
  // element's face. Can save storage if common bdry face data is used by face id?
  // Replace flat arrays of Write<> by convenient,
  // Kokkos::View<double[numAddedBdryFaces][SIZE_PER_FACE]> ..  ?

  o::Write<o::Real> bdryFacesCsrW(numAddedBdryFaces * SIZE_PER_FACE, 0);

  auto convert = OMEGA_H_LAMBDA(o::LO elem){
    // Index points to data counting spread out face or x,y,z components.
    // This calculates face number from the above. 
    auto beg = bdryFaceInds[elem] / SIZE_PER_FACE;
    auto end = bdryFaceInds[elem+1] / SIZE_PER_FACE;
    if(beg == end){
      return;
    }
    // Source position calculated using pre-assigned fixed number of faces
    // per element. Here face number (outer loop) is used to get absolute position.

    for(o::LO i = beg; i<end; ++i){ //memcopy ?
      // Copy as Real
      bdryFacesCsrW[i*SIZE_PER_FACE] = bdryFaceIds[i];
      for(o::LO j= 1; j<SIZE_PER_FACE+1; ++j){
        bdryFacesCsrW[i*SIZE_PER_FACE + j] = bdryFacesW[elem*i*SIZE_PER_FACE + j];
      }
    }    
  };
  o::parallel_for(nel, convert, "Convert to CSR write");
  //CSR data. Conversion is making RO object on device by passing device data.
  // No fence needed, since not copying to host.
  bdryFaces = o::Reals(bdryFacesCsrW);

}




// Pre-processing to fill bdry faces in elements near bdry within depth needed. 
// Call init function before this. Filling flat arrays; Later call convert2ReadOnlyCSR.

o::LO verbose = 0;
void GitrmMesh::copyBdryFacesToSelf() {

  MESHDATA(mesh);
  auto &bdryFacesW = this->bdryFacesW;
  auto &numBdryFaceIds = this->numBdryFaceIds;
  auto &bdryFaceIds = this->bdryFaceIds;
  auto &bdryFaceElemIds = this->bdryFaceElemIds;
  auto &bdryFlags = this->bdryFlags;

  auto fillBdry = OMEGA_H_LAMBDA(o::LO elem){

    // Checking bdry faces
    o::LO nbdry = 0; 
    o::LO bdryFids[4];
    const auto beg_face = elem *4;
    const auto end_face = beg_face +4;
    for(o::LO fi = beg_face; fi < end_face; ++fi){//not 0..3
      const auto fid = down_r2f[fi];
      if(verbose>2){
              printf( "exposed: e_%d, i_%d, fid_%d => %d\n", elem, fi, fid, side_is_exposed[fid]);
      }
      if(side_is_exposed[fid]) {
        bdryFids[nbdry] = fid;
        ++nbdry;
      }
    }
    // TODO: pisces mesh elem id 9 is exposed, but not as per side_is_exposed.
    // Check if the mesh created matches with the mesh data

    //If no bdry faces to add
    if(!nbdry){
      return;
    }
    if(verbose>2) printf("\nelem:%d, nbdry: %d \n", elem, nbdry);

    //Only boundary faces
    for(o::LO fi = 0; fi < nbdry; ++fi){
      auto fid = bdryFids[fi];
      auto fv2v = o::gather_verts<3>(face_verts, fid); //Few<LO, 3>
      const auto face = o::gather_vectors<3, 3>(coords, fv2v);
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
  auto &bdryFacesW = this->bdryFacesW;
  auto &numBdryFaceIds = this->numBdryFaceIds;
  auto &bdryFaceIds = this->bdryFaceIds;
  auto &bdryFaceElemIds = this->bdryFaceElemIds;
  auto &bdryFlags = this->bdryFlags;

  copyBdryFacesToSelf();

  auto fill = OMEGA_H_LAMBDA(o::LO elem){
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
    
    // Store existing # of faces
    o::LO existing = numBdryFaceIds[elem];
    o::LO tot = existing;
    const auto beg_face = elem *4;
    const auto end_face = beg_face +4;
    auto dface_ind = dual_elems[elem]; //first interior
    for(auto fi = beg_face; fi < end_face; ++fi){//not 0..3
      const auto fid = down_r2f[fi];
      if(side_is_exposed[fid]) {
        ++nbf;
        // Bdry element need not have regions that have shortest dist to bdry
        // always on the own bdry face. So, need to add adj. faces.
        continue;
      }

      //Check all adjacent elements
      auto adj_elem  = dual_faces[dface_ind];

      auto fv2v = o::gather_verts<3>(face_verts, fid); //Few<LO, 3>
      const auto face = o::gather_vectors<3, 3>(coords, fv2v);
      // Get ids updated by adj. elems of previous faces processed
      o::LO sizeAdj = numBdryFaceIds[adj_elem]; 

      if(verbose >2) 
        printf("\n **tot:%d , SizeAdj_of_%d = %d \n", tot, adj_elem, sizeAdj);

      for(o::LO ai=0; ai < sizeAdj; ++ai){
        bool found = false;
        auto index = BDRYFACE_SIZE*adj_elem + ai;
        auto adj_fid = bdryFaceIds[index];
        auto adj_fElid = bdryFaceElemIds[index];

         if(verbose >2) 
          printf("adj_elem_%d, fadj_ %d, of_e_%d : ", adj_elem,  adj_fid, adj_fElid); 

        for(auto fi = 0; fi < tot; ++fi){
          if(verbose >2) printf("   this_fid_ %d ", bdryFaceIds[BDRYFACE_SIZE*elem+fi]);
          if(adj_fid == bdryFaceIds[BDRYFACE_SIZE*elem+fi]){ 
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
        bool add = p::checkIfFaceWithinDistToTet(tet, face_verts, coords, 
          adj_fid, DEPTH_DIST2_BDRY);
        if(!add) {
          if(verbose >2) printf("NOT adding \n");
          continue;
        }       
        if(verbose >2) 
          printf("From_%d ADDING.. fid %d of_e_%d\n", adj_elem , adj_fid, adj_fElid);

        for(o::LO i=0; i<3; ++i){
          for(o::LO j=0; j<3; j++){
            bdryFacesW[elem* BDRYFACE_SIZE*(SIZE_PER_FACE-1) 
              + tot* (SIZE_PER_FACE-1) + i*3 + j] = face[i][j];
          }
        }
        auto ind = elem* BDRYFACE_SIZE + tot;
        bdryFaceIds[ind] = adj_fid;
        bdryFaceElemIds[ind] = adj_fElid;
        ++tot;

        if(verbose >2){ 
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
      for(o::LO i=0; i<4-nbf; ++i){
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
       KOKKOS_LAMBDA(const int& i, o::LO & lsum ) {
     lsum += bdryFlags[i];
   }, total);

   printf("LOOP %d total %d \n", ln, total);
   ++ln;
  }

}

void GitrmMesh::printBdryFaceIds(bool printIds=true, o::LO minNums=10){
  auto &numBdryFaceIds = this->numBdryFaceIds;
  auto &bdryFaceIds = this->bdryFaceIds;
  auto &bdryFaceElemIds = this->bdryFaceElemIds;
  auto print = OMEGA_H_LAMBDA(o::LO elem){
    o::LO nb = numBdryFaceIds[elem];
    if(nb > minNums){
      printf(" %d (tot %d ) :: ", elem, numBdryFaceIds[elem]);
      if(printIds){
        for(o::LO i=0; i<nb; ++i){
          o::LO ind = elem* BDRYFACE_SIZE+i;
          printf("%d( %d ), ", bdryFaceIds[ind], bdryFaceElemIds[ind]);
        }
      }
      printf("\n");
    }
  };
  
  o::LO nums = mesh.nelems();
  o::parallel_for(nums, print, "printAFewBdryFaceNums");
  printf("\nDEPTH: %0.5f \n", DEPTH_DIST2_BDRY);
}


