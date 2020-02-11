#include <fstream>
#include <cstdlib>
#include <vector>
#include <set>
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
//#include "Omega_h_library.hpp"
//#include "pumipic_mesh.hpp"

#include "GitrmInputOutput.hpp"

int iTimePlusOne = 0;

//TODO remove mesh argument, once Singleton gm is used
GitrmParticles::GitrmParticles(o::Mesh& m, double dT):
  ptcls(nullptr), mesh(m), timeStep(dT)
{}

GitrmParticles::~GitrmParticles() {
  delete ptcls;
}

// Initialized in only one element
void GitrmParticles::defineParticles(p::Mesh& picparts, int numPtcls, 
  o::LOs& ptclsInElem, int elId) {
  o::Int ne = mesh.nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = mesh_element_gids[i];
  });
  if(elId>=0) {
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
      ptcls_per_elem(i) = 0;
      if (i == elId) {
        ptcls_per_elem(i) = numPtcls;
        printf(" Ptcls in elId %d\n", elId);
      }
    });
  } else {
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
      ptcls_per_elem(i) = ptclsInElem[i];
    });
  }

  printf(" ptcls/elem: \n");
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
    if (np > 0)
      printf("%d , ", np);
  });
  printf("\n");
  //'sigma', 'V', and the 'policy' control the layout of the PS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = 1; //INT_MAX; // full sorting
  const int V = 128;//1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  printf("Constructing Particles\n");
  //Create the particle structure
  ptcls = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
                   ptcls_per_elem, element_gids);
}

void GitrmParticles::initPtclsFromFile(p::Mesh& picparts, 
  const std::string& fName,  o::LO& numPtcls, o::LO maxLoops, bool printSource) {
  std::cout << "Loading particle initial data from file: " << fName << " \n";
  o::HostWrite<o::Real> readInData_h;
  // TODO piscesLowFlux/updated/input/particleSource.cfg has r,z,angles, CDF, cylSymm=1
  int numPtclsRead;
  auto stat = readParticleSourceNcFile(fName, readInData_h, numPtcls, 
    numPtclsRead, true);

  OMEGA_H_CHECK((!stat) && (numPtclsRead > 0));  
  OMEGA_H_CHECK((numPtcls > 0) && (mesh.nelems() >0));
  o::Reals readInData_r(readInData_h);
  o::LOs elemIdOfPtcls;
  o::LOs numPtclsInElems;
  std::cout << "findElemIdsOfPtclFileCoordsByAdjSearch \n";
  findElemIdsOfPtclFileCoordsByAdjSearch(readInData_r, elemIdOfPtcls,
    numPtclsInElems, numPtcls, numPtclsRead);

  printf("Constructing PS particles\n");
  defineParticles(picparts, numPtcls, numPtclsInElems, -1);
  
  initPtclWallCollisionData(numPtcls);
  //note:rebuild to get mask if elem_ids changed
  printf("Setting ImpurityPtcl InitCoords \n");
  o::LOs ptclIdPtrsOfElem;
  o::LOs ptclIdsInElem;
  printf("calculating CSR indices\n");
  auto totalPtcls = calculateCsrIndices(numPtclsInElems, ptclIdPtrsOfElem);
  OMEGA_H_CHECK(numPtcls == totalPtcls);

  printf("converting  to CSR\n");
  convertInitPtclElemIdsToCSR(numPtclsInElems, ptclIdPtrsOfElem, 
    ptclIdsInElem, elemIdOfPtcls, numPtcls);
  printf("setting ptcl Ids \n");
  auto mesh = picparts.mesh();
  setPidsOfPtclsLoadedFromFile(ptclIdPtrsOfElem, ptclIdsInElem, 
    elemIdOfPtcls, numPtcls, mesh->nelems());
  printf("setting ptcl init data \n");
  setPtclInitData(readInData_r, numPtclsRead);
  printf("setting ionization recombination init data \n");
  initPtclChargeIoniRecombData();
  initPtclSurfaceModelData();

  if(printSource)
    printPtclSource(readInData_r, numPtcls, 6); //nptcl=0(all), dof=6
}

void GitrmParticles::initPtclWallCollisionData(int numPtcls) {
  wallCollisionPts = o::Write<o::Real>(3*numPtcls, 0, "xpoints");
  wallCollisionFaceIds = o::Write<o::LO>(numPtcls, -1);
}


// Find elemId of any particle, and start with that elem to search 
// elem of all particles. Get #particles in each element,
// for PS_LAMBDA to fill ptcl data in ptcls.
void GitrmParticles::findElemIdsOfPtclFileCoordsByAdjSearch( 
  const o::Reals& data, o::LOs& elemIdOfPtcls, o::LOs& numPtclsInElems,
  o::LO numPtcls, o::LO numPtclsRead) {
  o::LO maxLoop = 10000;
  MESHDATA(mesh);
  auto size = data.size();
  o::Write<o::LO> elemDet(1, -1);
  // Beginning element id of this x,y,z
  o::LO elmBeg=-1, ii=0;
  bool found = false;
  while(!found) {
    auto lamb = OMEGA_H_LAMBDA(const int elem) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);
      auto pos = o::zero_vector<3>();
      for(int j=0; j<3; ++j)
        pos[j] = data[j*numPtclsRead+ii];
      auto bcc = o::zero_vector<4>();
      p::find_barycentric_tet(M, pos, bcc);
      if(p::all_positive(bcc, 1.0e-6)) {
        elemDet[0] = elem;
      }
    };
    o::parallel_for(nel, lamb, "search_parent_of_ptcl1");
    o::HostRead<o::LO> elemFound(elemDet);
    elmBeg = elemFound[0];
    if(elmBeg >= 0 || ii > maxLoop)
      break;
    ++ii;
  }

  printf(" ELEM_beg %d of_ptcl %d of %d \n", elmBeg, ii, numPtcls);
  if(elmBeg < 0)
    Omega_h_fail("Failed finding initial element \n");
  o::Write<o::LO> numPtclsInElems_w(nel, 0);
  o::Write<o::LO> elemIdOfPtcls_w(numPtcls, -1);
  o::Write<o::LO> ptcl_done(numPtcls, 0);
  o::LO maxSearch = 100;
  //search all particles starting with this element
  auto lamb2 = OMEGA_H_LAMBDA(const int ip) {
    bool found = false;
    auto pos = o::zero_vector<3>();
    auto bcc = o::zero_vector<4>();
    o::LO elem = elmBeg, isearch=0;
    while(!found) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);
      for(int j=0; j<3; ++j)
        pos[j] = data[j*numPtclsRead+ip];
      p::find_barycentric_tet(M, pos, bcc);
      if(p::all_positive(bcc, 0)) {
        elemIdOfPtcls_w[ip] = elem;
        ptcl_done[ip] = 1;
        Kokkos::atomic_increment(&numPtclsInElems_w[elem]);
        found = true;
      } else {
        o::LO minInd = p::min_index(bcc, 4);
        auto dual_elem_id = dual_faces[elem];
        o::LO findex = 0;
        for(auto iface = elem*4; iface < (elem+1)*4; ++iface) {
          auto face_id = down_r2fs[iface];
          bool exposed = side_is_exposed[face_id];
          if(!exposed) {
            if(findex == minInd)
              elem = dual_elems[dual_elem_id];
            ++dual_elem_id;
          }
          ++findex;
        }//for
      }
      if(isearch > maxSearch)
        break;
      ++isearch;
    }
  };
  o::parallel_for(numPtcls, lamb2, "init_impurity_ptcl2");
  o::LOs ptcl_done_r(ptcl_done);
  auto minFlag = o::get_min(ptcl_done_r);
  if(!minFlag) {
    o::parallel_for(numPtcls, OMEGA_H_LAMBDA(const int i) {
      if(!ptcl_done[i]) {
        double v[6];
        for(int j=0; j<6; ++j)
          v[j] = data[j*numPtclsRead+i];      
        printf("NOTdet i %d %g %g %g :vel: %g %g %g\n", 
          i, v[0], v[1], v[2], v[3], v[4], v[5] );
      }
    });
  }

  OMEGA_H_CHECK(minFlag == 1);
  numPtclsInElems = o::LOs(numPtclsInElems_w);
  elemIdOfPtcls = o::LOs(elemIdOfPtcls_w); 
}

// using ptcl sequential numbers 0..numPtcls
void GitrmParticles::convertInitPtclElemIdsToCSR(const o::LOs& numPtclsInElems,
  o::LOs& ptclIdPtrsOfElem, o::LOs& ptclIdsInElem, o::LOs& elemIdOfPtcls,
  o::LO numPtcls) {
  o::LO debug = 0;
  auto nel = mesh.nelems();
  // csr data
  o::Write<o::LO> ptclIdsInElem_w(numPtcls, -1);
  o::Write<o::LO> ptclsFilledInElem(nel, 0); 
  auto lambda = OMEGA_H_LAMBDA(const o::LO id) {
    auto el = elemIdOfPtcls[id];
    auto old = Kokkos::atomic_fetch_add(&(ptclsFilledInElem[el]), 1);
    //TODO FIXME invalid device function error with OMEGA_H_CHECK in lambda
    auto nLimit = numPtclsInElems[el];
    OMEGA_H_CHECK(old < nLimit); 
    //elemId is sequential from 0 .. nel
    auto beg = ptclIdPtrsOfElem[el];
    auto pos = beg + old;
    auto idLimit = ptclIdPtrsOfElem[el+1];
    //FIXME same error as above here
    OMEGA_H_CHECK(pos < idLimit);
    auto prev = Kokkos::atomic_exchange(&(ptclIdsInElem_w[pos]), id);
    if(debug)
      printf("id:el %d %d old %d beg %d pos %d previd %d maxPtcls %d \n",
        id, el, old, beg, pos, prev, numPtclsInElems[el] );
  };
  o::parallel_for(numPtcls, lambda, "Convert to CSR write");
  ptclIdsInElem = o::LOs(ptclIdsInElem_w);   
}


void GitrmParticles::setPidsOfPtclsLoadedFromFile(const o::LOs& ptclIdPtrsOfElem,
  const o::LOs& ptclIdsInElem,  const o::LOs& elemIdOfPtcls, 
  const o::LO numPtcls, const o::LO nel) {
  int debug = 1;
  o::Write<o::LO> nextPtclInd(nel, 0);
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      auto nextInd = Kokkos::atomic_fetch_add(&(nextPtclInd[elem]), 1);
      auto ind = ptclIdPtrsOfElem[elem] + nextInd;
      auto limit = ptclIdPtrsOfElem[elem+1];
      //Set checks separately to avoid possible Error 
      OMEGA_H_CHECK(ind >= 0);
      OMEGA_H_CHECK(ind < limit);
      const auto ip = ptclIdsInElem[ind]; //ip 0..numPtcls
      if(debug && (ind < 0 || ind>= limit || ip<0 || ip>= numPtcls || elemIdOfPtcls[ip] != elem))
        printf("**** elem %d pid %d ind %d nextInd %d indlim %d ip %d e_of_p %d\n", 
          elem, pid, ind, nextInd, ptclIdPtrsOfElem[elem+1], ip, elemIdOfPtcls[ip]);
      OMEGA_H_CHECK(ip >= 0);
      OMEGA_H_CHECK(ip < numPtcls);
      OMEGA_H_CHECK(elemIdOfPtcls[ip] == elem);
      pid_ps(pid) = ip;
    }
  };
  ps::parallel_for(ptcls, lambda, "setPidsOfPtcls");
}

//To use ptcls, PS_LAMBDA is required, not parallel_for(#ptcls,OMEGA_H_LAMBDA
// since ptcls in each element is iterated in groups. Construct PS with 
// #particles in each elem passed in, otherwise newly added particles in 
// originally empty elements won't show up in PS_LAMBDA iterations. 
// ie. their mask will be 0. If mask is not used, invalid particles may 
// show up from other threads in the launch group.
void GitrmParticles::setPtclInitData(const o::Reals& data, int numPtclsRead) {
  o::LO debug = 0;
  const auto coords = mesh.coords(); 
  const auto mesh2verts = mesh.ask_elem_verts(); 
  auto next_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto pos_ps_d = ptcls->get<PTCL_POS>();
  auto vel_d = ptcls->get<PTCL_VEL>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);
      auto bcc = o::zero_vector<4>();
      auto vel = o::zero_vector<3>();
      auto pos = o::zero_vector<3>();
      auto ip = pid_ps(pid); 
      for(int i=0; i<3; ++i)
        pos[i] = data[i*numPtclsRead+ip];
      for(int i=0; i<3; ++i)
        vel[i] = data[(3+i)*numPtclsRead+ip];
      if(debug)
        printf("ip %d pos %g %g %g vel %g %g %g\n", ip, pos[0], pos[1], pos[2], 
          vel[0], vel[1], vel[2]);
      p::find_barycentric_tet(M, pos, bcc);
      OMEGA_H_CHECK(p::all_positive(bcc, 0));
      for(int i=0; i<3; i++) {
        pos_ps_d(pid,i) = pos[i];
        vel_d(pid, i) = vel[i];
        next_ps_d(pid,i) = 0;
      }
    }
  };
  ps::parallel_for(ptcls, lambda, "setPtclInitData");
}

void GitrmParticles::initPtclChargeIoniRecombData() {
  auto charge_ps = ptcls->get<PTCL_CHARGE>();
  auto first_ionizeZ_ps = ptcls->get<PTCL_FIRST_IONIZEZ>();
  auto prev_ionize_ps = ptcls->get<PTCL_PREV_IONIZE>();
  auto first_ionizeT_ps = ptcls->get<PTCL_FIRST_IONIZET>();
  auto prev_recomb_ps = ptcls->get<PTCL_PREV_RECOMBINE>();

  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask > 0) {
      charge_ps(pid) = 0;
      first_ionizeZ_ps(pid) = 0;
      prev_ionize_ps(pid) = 0;
      first_ionizeT_ps(pid) = 0;
      prev_recomb_ps(pid) = 0;
    }
  };
  ps::parallel_for(ptcls, lambda, "initPtclChargeIoniRecombData");
}

void GitrmParticles::initPtclSurfaceModelData() {
  auto ps_weight = ptcls->get<PTCL_WEIGHT>();
  auto ps_hitNum = ptcls->get<PTCL_HIT_NUM>();
  auto ps_newVelMag = ptcls->get<PTCL_VMAG_NEW>();
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask > 0) {
      ps_weight(pid) = 1;
      ps_hitNum(pid) = 0;
      ps_newVelMag(pid) = 0;
    }
  };
  ps::parallel_for(ptcls, lambda, "initPtclSurfaceModelData");
}

void GitrmParticles::initPtclsInADirection(p::Mesh& picparts, o::LO numPtcls, 
   o::Real theta, o::Real phi, o::Real r, o::LO maxLoops, o::Real outer) {
  o::Write<o::LO> elemAndFace(3, -1); 
  o::LO initEl = -1;
  findInitialBdryElemIdInADir(theta, phi, r, initEl, elemAndFace, maxLoops, outer);
  o::LOs temp;
  defineParticles(picparts, numPtcls, temp, initEl);
  printf("Constructed Particles\n");

  //note:rebuild if particles to be added in new elems, or after emptying any elem.
  printf("\n Setting ImpurityPtcl InitCoords \n");
  setPtclInitRndDistribution(elemAndFace);
}

void GitrmParticles::setPtclInitRndDistribution(o::Write<o::LO> &elemAndFace) {
  MESHDATA(mesh);

  //Set particle coordinates. Initialized only on one face. TODO confirm this ? 
  auto x_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto x_ps_prev_d = ptcls->get<PTCL_POS>();
  auto vel_d = ptcls->get<PTCL_VEL>();
  int psCapacity = ptcls->capacity();
  o::HostWrite<o::Real> rnd1(psCapacity, 0);
  o::HostWrite<o::Real> rnd2(psCapacity, 0);
  o::HostWrite<o::Real> rnd3(psCapacity, 0);
  std::srand(time(NULL));
  for(auto i=0; i<psCapacity; ++i) {
    rnd1[i] = (o::Real)(std::rand())/RAND_MAX;
    rnd2[i] = (o::Real)(std::rand())/RAND_MAX;
    rnd3[i] = (o::Real)(std::rand())/RAND_MAX;
  }
  o::Reals rand1 = o::Reals(o::Write<o::Real>(rnd1));
  o::Reals rand2 = o::Reals(o::Write<o::Real>(rnd2));
  o::Reals rand3 = o::Reals(o::Write<o::Real>(rnd3));

  o::Write<o::LO> elem_ids(psCapacity,-1);
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
    //if(elemAndFace[1] >=0 && elem == elemAndFace[1]) {  
      o::LO verbose =1;
      // TODO if more than an element  ?
      const auto faceId = elemAndFace[2];
      const auto fv2v = o::gather_verts<3>(face_verts, faceId);
      const auto face = p::gatherVectors3x3(coords, fv2v);
      auto fcent = p::face_centroid_of_tet(faceId, coords, face_verts);
      auto tcent = p::centroid_of_tet(elem, mesh2verts, coords); 
      auto diff = tcent - fcent;
      if(verbose >3)
        printf(" elemAndFace[1]:%d, elem:%d face%d beg%d\n", 
          elemAndFace[1], elem, elemAndFace[2], elemAndFace[0]);

      o::Vector<4> bcc;
      o::Vector<3> pos;
      auto rn1 = rand1[pid];
      auto rn2 = rand2[pid];
      do { 
        o::Real bc1 = (rn1 > rn2) ? rn2: rn1;
        o::Real bc2 = std::abs(rn1 - rn2);
        o::Real bc3 = 1.0 - bc1 - bc2;
        o::Vector<3> fpos = bc1*face[0] + bc2*face[1] + bc3*face[2];
        auto fnorm = p::face_normal_of_tet(faceId, elem, coords, mesh2verts, 
                                        face_verts, down_r2fs);
        pos = fpos - 1.0e-6*fnorm;
        auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
        auto M = p::gatherVectors4x3(coords, tetv2v);
        p::find_barycentric_tet(M, pos, bcc);
        rn1 /= 2.0;
      } while(!p::all_positive(bcc, 0));

      double amu = 2.0; //TODO
      double energy[] = {4.0, 4, 4}; //TODO actual [4,0,0]
      double vel[] = {0,0,0};
      for(int i=0; i<3; i++) {
        x_ps_prev_d(pid,i) = pos[i];
        x_ps_d(pid,i) = pos[i];
        auto en = energy[i];   
        //if(! o::are_close(energy[i], 0))
        vel[i] = std::sqrt(2.0 * abs(en) * 1.60217662e-19 / (amu * 1.6737236e-27));
        vel[i] *= rand3[pid];  //TODO
      }

      for(int i=0; i<3; i++)
        vel_d(pid, i) = vel[i];
 
      elem_ids[pid] = elem;

      if(verbose >2)
        printf("elm %d : pos %.4f %.4f %.4f : vel %.1f %.1f %.1f Mask%d\n",
          elem, x_ps_prev_d(pid,0), x_ps_prev_d(pid,1), x_ps_prev_d(pid,2),
          vel[0], vel[1], vel[2], mask);
    }
  };
  ps::parallel_for(ptcls, lambda, "setPtclInitRndDistribution");
}

// spherical coordinates (wikipedia), radius r=1.5m, inclination theta[0,pi] from the z dir,
// azimuth angle phi[0, 2π) from the Cartesian x-axis (so that the y-axis has phi = +90°).
void GitrmParticles::findInitialBdryElemIdInADir(o::Real theta, o::Real phi, o::Real r,
     o::LO &initEl, o::Write<o::LO> &elemAndFace, o::LO maxLoops, o::Real outer){

  o::LO debug = 4;
  MESHDATA(mesh);

  theta = theta * o::PI / 180.0;
  phi = phi * o::PI / 180.0;
  
  const o::Real x = r * sin(theta) * cos(phi);
  const o::Real y = r * sin(theta) * sin(phi);
  const o::Real z = r * cos(theta);

  o::Real endR = r + outer; //meter, to be outside of the domain
  const o::Real xe = endR * sin(theta) * cos(phi);
  const o::Real ye = endR * sin(theta) * sin(phi);
  const o::Real ze = endR * cos(theta);

  printf("\nDirection:x,y,z: %f %f %f\n xe,ye,ze: %f %f %f\n", x,y,z, xe,ye,ze);

  // Beginning element id of this x,y,z
  auto lamb = OMEGA_H_LAMBDA(const int elem) {
    auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
    auto M = p::gatherVectors4x3(coords, tetv2v);

    o::Vector<3> orig;
    orig[0] = x;
    orig[1] = y;
    orig[2] = z;
    o::Vector<4> bcc;
    p::find_barycentric_tet(M, orig, bcc);
    if(p::all_positive(bcc, 0)) {
      elemAndFace[0] = elem;
      if(debug > 3)
        printf(" ORIGIN detected in elem %d \n", elem);
    }
  };
  o::parallel_for(mesh.nelems(), lamb, "init_impurity_ptcl1");
  o::HostRead<o::LO> elemId_bh(elemAndFace);
  printf(" ELEM_beg %d \n", elemId_bh[0]);
  
  if(elemId_bh[0] < 0) {
    Omega_h_fail("Failed finding initial element in given direction\n");
  }

  // Search final elemAndFace on bdry, on 1 thread on device(issue [] on host) 
  o::Write<o::Real> xpt(3, -1); 
  auto lamb2 = OMEGA_H_LAMBDA(const int e) {
    auto elem = elemAndFace[0];
    o::Vector<3> dest;
    dest[0] = xe;
    dest[1] = ye;
    dest[2] = ze;
    o::Vector<3> orig;
    orig[0] = x;
    orig[1] = y;
    orig[2] = z;    
    o::Vector<4> bcc;
    bool found = false;
    o::LO loops = 0;

    while (!found) {

      if(debug > 4)
        printf("\n****ELEM %d : ", elem);

      // Destination should be outisde domain
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);

      p::find_barycentric_tet(M, dest, bcc);
      if(p::all_positive(bcc, 0)) {
        printf("Wrong guess of destination in initImpurityPtcls");
        OMEGA_H_CHECK(false);
      }

      // Start search
      auto dual_elem_id = dual_faces[elem];
      const auto beg_face = elem *4;
      const auto end_face = beg_face +4;
      o::LO fIndex = 0;

      for(auto iface = beg_face; iface < end_face; ++iface) {
        const auto face_id = down_r2fs[iface];

        o::Vector<3> xpoint = o::zero_vector<3>();
        const auto face = p::get_face_coords_of_tet(mesh2verts, coords, elem, fIndex);
        o::Real dproj = 0;
        bool detected = p::line_triangle_intx_simple(face, orig, dest, xpoint, dproj);
        if(debug > 4) {
          printf("iface %d faceid %d detected %d\n", iface, face_id, detected);             
        }

        if(detected && side_is_exposed[face_id]) {
          found = true;
          elemAndFace[1] = elem;
          elemAndFace[2] = face_id;

          for(o::LO i=0; i<3; ++i)
            xpt[i] = xpoint[i];

          if(debug) {
            printf(" faceid %d detected on exposed\n",  face_id);
          }
          break;
        } else if(detected && !side_is_exposed[face_id]) {
          auto adj_elem  = dual_elems[dual_elem_id];
          elem = adj_elem;
          if(debug >4) {
            printf(" faceid %d detected on interior; next elm %d\n", face_id, elem);
          }
          break;
        }
        if(!side_is_exposed[face_id]){
          ++dual_elem_id;
        }
        ++fIndex;
      } // faces

      if(loops > maxLoops) {
          printf("Tried maxLoops iterations in initImpurityPtcls");
          OMEGA_H_CHECK(false);
      }
      ++loops;
    }
  };
  o::parallel_for(1, lamb2, "init_impurity_ptcl2");

  o::HostRead<o::Real> xpt_h(xpt);

  o::HostRead<o::LO> elemId_fh(elemAndFace);
  initEl = elemId_fh[1];
  printf(" ELEM_final %d xpt: %.3f %.3f %.3f\n\n", elemId_fh[1], xpt_h[0], xpt_h[1], xpt_h[2]);
  OMEGA_H_CHECK((initEl>=0) && (elemId_fh[0]>=0));
}

// Read GITR particle step data of all time steps; eg: rand numbers.
int GitrmParticles::readGITRPtclStepDataNcFile(const std::string& ncFileName, 
  int& maxNPtcls, int& numPtclsRead, bool debug) {
  assert(USE_GITR_RND_NUMS == 1);
  std::cout << "Reading Test GITR step data " << ncFileName << "\n";
  // re-order the list in its constructor to leave out empty {}
  Field3StructInput fs({"intermediate"}, {}, {"nP", "nTHist", "dof"}, 0,
    {"RndIoni_at", "RndRecomb_at", "RndCollision_n1_at", "RndCollision_n2_at", 
     "RndCollision_xsi_at", "RndCrossField_at", "RndReflection_at",
     "Opt_IoniRecomb", "Opt_Diffusion", "Opt_Collision", "Opt_SurfaceModel"}); 
  auto stat = readInputDataNcFileFS3(ncFileName, fs, maxNPtcls, numPtclsRead, 
      "nP", debug);
  testGitrPtclStepData = o::Reals(fs.data);
  testGitrDataIoniRandInd = fs.getIntValueOf("RndIoni_at");
  testGitrDataRecRandInd = fs.getIntValueOf("RndRecomb_at");
  testGitrStepDataDof = fs.getIntValueOf("dof"); // or fs.getNumGrids(2);
  testGitrStepDataNumTsteps = fs.getIntValueOf("nTHist") - 1; // NOTE
  testGitrStepDataNumPtcls = fs.getIntValueOf("nP");
  testGitrCrossFieldDiffRndInd = fs.getIntValueOf("RndCrossField_at");
  testGitrCollisionRndn1Ind = fs.getIntValueOf("RndCollision_n1_at"); 
  testGitrCollisionRndn2Ind = fs.getIntValueOf("RndCollision_n2_at");
  testGitrCollisionRndxsiInd = fs.getIntValueOf("RndCollision_xsi_at");
  testGitrReflectionRndInd = fs.getIntValueOf("RndReflection_at");
  testGitrOptIoniRec = fs.getIntValueOf("Opt_IoniRecomb");
  testGitrOptDiffusion =  fs.getIntValueOf("Opt_Diffusion");
  testGitrOptCollision =  fs.getIntValueOf("Opt_Collision");
  testGitrOptSurfaceModel = fs.getIntValueOf("Opt_SurfaceModel");

  //if(debug)
    printf(" TestGITRdata: dof %d nT %d nP %d Index: rndIoni %d rndRec %d \n"
      " rndCrossFieldDiff %d rndColl_n1 %d rndColl_n2 %d rndColl_xsi %d "
      " rndReflection %d\n GITR run Flags: ioniRec %d diffusion %d "
      " collision %d surfmodel %d \n", testGitrStepDataDof, testGitrStepDataNumTsteps,
      testGitrStepDataNumPtcls, testGitrDataIoniRandInd, testGitrDataRecRandInd, 
      testGitrCrossFieldDiffRndInd, testGitrCollisionRndn1Ind, 
      testGitrCollisionRndn2Ind, testGitrCollisionRndxsiInd, 
      testGitrReflectionRndInd, testGitrOptIoniRec, testGitrOptDiffusion,
      testGitrOptCollision, testGitrOptSurfaceModel);

  return stat;
}

//timestep >0
void GitrmParticles::checkCompatibilityWithGITRflags(int timestep) {
  if(timestep==0) 
    printf("ERROR: checkCompatibility is done before variables set\n");
  OMEGA_H_CHECK(timestep>0);
  if(ranIonization||ranRecombination)
    OMEGA_H_CHECK(testGitrOptIoniRec);
  else
    OMEGA_H_CHECK(!testGitrOptIoniRec);

  if(ranCoulombCollision)
    OMEGA_H_CHECK(testGitrOptCollision);
  else
    OMEGA_H_CHECK(!testGitrOptCollision);

  if(ranDiffusion)
    OMEGA_H_CHECK(testGitrOptDiffusion);
  else
    OMEGA_H_CHECK(!testGitrOptDiffusion);
  if(ranSurfaceReflection)
    OMEGA_H_CHECK(testGitrOptSurfaceModel);
  else
    OMEGA_H_CHECK(!testGitrOptSurfaceModel);
}


void printPtclSource(o::Reals& data, int nPtcls, int numPtclsRead) {
  o::HostRead<o::Real>dh(data);
  printf("ParticleSourcePositions: nptcl= %d\n", nPtcls);
  for(int ip=0; ip<nPtcls; ++ip) {
    printf("PtclSource-pos-vel ");
    for(int j=0; j<3; ++j)
      printf("%g ", dh[j*numPtclsRead+ip]);
    for(int j=3; j<6; ++j)
      printf("%g ", dh[j*numPtclsRead+ip]);
    printf("\n");
  }
  printf("\n");
}

void printStepData(std::ofstream& ofsHistory, PS* ptcls, int iter, 
  int numPtcls, o::Write<o::Real>& ptclsDataAll,
  o::Write<o::LO>& lastFilledTimeSteps, o::Write<o::Real>& data, 
  int dof, bool accum) {
  if(iter ==0)
    updatePtclStepData(ptcls, ptclsDataAll,lastFilledTimeSteps, numPtcls, dof);

  if(accum) {
    o::HostWrite<o::Real> ptclAllHost(ptclsDataAll);
    printPtclHostData(ptclAllHost, ofsHistory, numPtcls, dof, 
      "ptclHistory_accum", iter);
  } else {
    o::HostWrite<o::Real> dh(data);
    o::HostWrite<o::Real> dp(ptcls->nPtcls() * dof);
    for(int ip=0, n=0; ip< numPtcls; ++ip) {
      //only ptcls from ptcls->nptcls have valid ids
      int id = static_cast<int>(dh[ip*dof+6]);
      if(id < 0)
        continue;
      for(int i=0; i<8; ++i)
        dp[n*dof+i] = dh[id*dof+i];
      ++n;
    }
    printPtclHostData(dp, ofsHistory, ptcls->nPtcls(), dof, "ptclHistory", iter);
  }
}

void writePtclStepHistoryFile(o::Write<o::Real>& ptclsHistoryData, 
  o::Write<o::LO>& lastFilledTimeSteps, int numPtcls, int dof, 
  int nTHistory, std::string outNcFileName) {
  
  //fill empty elements with last filled values
  auto lambda = OMEGA_H_LAMBDA(const int& pid) {
    auto ts = lastFilledTimeSteps[pid];
    for(int idof=0; idof<dof; ++idof) { 
      auto ref = ts*numPtcls*dof + pid*dof + idof;
      auto dat = ptclsHistoryData[ref];
      for(int it = ts+1; it < nTHistory; ++it) {
        auto ind = it*numPtcls*dof + pid*dof + idof;
        ptclsHistoryData[ind] = dat;
      }
    }
  };
  o::parallel_for(numPtcls, lambda);
  OutputNcFileFieldStruct outStruct({"nP", "nT"}, {"x", "y", "z", "vx", "vy", "vz"},
                                    {numPtcls, nTHistory});
  writeOutputNcFile(ptclsHistoryData, numPtcls, dof, outStruct, outNcFileName);
}

void updatePtclStepData(PS* ptcls, o::Write<o::Real>& ptclStepData, 
   o::Write<o::LO>& lastFilledTimeSteps, int numPtcls, int dof, int iHistStep) { 
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto step = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto id = pid_ps(pid);
      auto vel = p::makeVector3(pid, vel_ps);
      auto pos = p::makeVector3(pid, pos_ps);
      lastFilledTimeSteps[id] = iHistStep;
      int beg = numPtcls*dof*iHistStep + id*dof; //storage format
      for(int i=0; i<3; ++i) {
        ptclStepData[beg+i] = pos[i];
        ptclStepData[beg+3+i] = vel[i];
      }
      //NOTE: many of device prints go missing
    }// mask
  };
  ps::parallel_for(ptcls, step, "updateStepData");
}

