#include <fstream>
#include <cstdlib>
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "Omega_h_library.hpp"


//TODO remove mesh argument, once Singleton gm is used
GitrmParticles::GitrmParticles(o::Mesh &m):
  mesh(m) {
}

GitrmParticles::~GitrmParticles() {
  delete scs;
}

// Initialized in only one element
void GitrmParticles::defineParticles(int numPtcls, o::LOs& ptclsInElem, int elId) {

  o::Int ne = mesh.nelems();
  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  //Element gids is left empty since there is no partitioning of the mesh yet
  SCS::kkGidView element_gids("elem_gids", ne);
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

  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
    if (np > 0)
      printf(" ptcls/elem[%d] %d\n", i, np);
    
    //TODO fix this for mpi
    //element_gids(i) = i; 
  });
  //'sigma', 'V', and the 'policy' control the layout of the SCS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  printf("Constructing Particles\n");
  //Create the particle structure
  scs = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
                   ptcls_per_elem, element_gids);
}

void GitrmParticles::initImpurityPtclsInADir(o::Real dTime, o::LO numPtcls, 
   o::Real theta, o::Real phi, o::Real r, o::LO maxLoops, o::Real outer) {
  o::Write<o::LO> elemAndFace(3, -1); 
  o::LO initEl = -1;
  findInitialBdryElemIdInADir(theta, phi, r, initEl, elemAndFace, maxLoops, outer);
  o::LOs temp;
  defineParticles(numPtcls, temp, initEl);
  printf("Constructed Particles\n");

  //note:rebuild if particles to be added in new elems, or after emptying any elem.
  printf("\n Setting ImpurityPtcl InitCoords \n");
  setImpurityPtclInitRndDistribution(elemAndFace);
}


void GitrmParticles::initImpurityPtclsFromFile(const std::string& fName, 
  o::LO numPtcls, o::LO maxLoops) {

  std::cout << "Loading particle initial data from file: " << fName << " \n";
  o::HostWrite<o::Real> readInData;
  // TODO piscesLowFlux/updated/input/particleSource.cfg has r,z,angles, CDF, cylSymm=1
  PtclInitStruct psin("ptcl_init_data", "nP", "x", "y", "z", "vx", "vy", "vz");
  processPtclInitFile(fName, readInData, psin, numPtcls);
  numPtcls = psin.nP;
  OMEGA_H_CHECK(numPtcls > 0 && mesh.nelems() >0);
  
  o::LOs elemIdOfPtcls;
  o::LOs numPtclsInElems;
  o::Reals readInData_r(readInData);
  std::cout << "findElemIdsOfPtclFileCoordsByAdjSearch \n";
  findElemIdsOfPtclFileCoordsByAdjSearch(numPtcls, readInData_r, elemIdOfPtcls,
   numPtclsInElems);

  printf("Constructing SCS particles\n");
  defineParticles(numPtcls, numPtclsInElems, -1);

  //note:rebuild to get mask if elem_ids changed
  printf("\n Setting ImpurityPtcl InitCoords \n");
  o::LOs ptclIdPtrsOfElem;
  o::LOs ptclIdsInElem;
  
  convertInitPtclElemIdsToCSR(numPtclsInElems, ptclIdPtrsOfElem, 
    ptclIdsInElem, elemIdOfPtcls, numPtcls);
  setImpurityPtclInitData(numPtcls, readInData_r, ptclIdPtrsOfElem, 
    ptclIdsInElem, elemIdOfPtcls, maxLoops);
}

// Find elemId of any particle, and start with that elem to search 
// elem of all particles. Get #particles in each element,
// for SCS_LAMBDA to fill ptcl data in scs.
void GitrmParticles::findElemIdsOfPtclFileCoordsByAdjSearch(o::LO numPtcls, 
  const o::Reals& data_r, o::LOs& elemIdOfPtcls, o::LOs& numPtclsInElems) {
  o::LO debug =1;
  o::LO maxLoop = 10;
  auto ne = mesh.nelems();
  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto down_r2fs = down_r2f.ab2b;
  const auto dual_faces = dual.ab2b;
  const auto dual_elems = dual.a2ab;
  
  auto size = data_r.size();
  auto dof = numPtcls/size;
  //TODO
  dof = 6;
  o::Write<o::LO> elemDet(1, -1);
  // Beginning element id of this x,y,z
  o::LO elmBeg=-1, ii=0;
  bool found = false;
  while(!found) {
    auto lamb = OMEGA_H_LAMBDA(const int elem) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);

      o::Vector<3> pos;
      pos[0] = data_r[ii*dof];
      pos[1] = data_r[ii*dof + 1];
      pos[2] = data_r[ii*dof + 2];
      o::Vector<4> bcc;
      p::find_barycentric_tet(M, pos, bcc);
      if(p::all_positive(bcc, 0)) {
        elemDet[0] = elem;
        if(debug > 3)
          printf(" ORIGIN detected in elem %d \n", elem);
      }
    };
    o::parallel_for(ne, lamb, "init_impurity_ptcl1");
  
    o::HostRead<o::LO> elemFound(elemDet);
    elmBeg = elemFound[0];
    if(elmBeg >= 0 || ii > maxLoop)
      break;
    ++ii;
  }

  printf(" ELEM_beg %d of_ptcl %d\n", elmBeg, ii);
  if(elmBeg < 0)
    Omega_h_fail("Failed finding initial element \n");
  o::Write<o::LO> numPtclsInElems_w(ne, 0);
  o::Write<o::LO> elemIdOfPtcls_w(numPtcls, -1);
  o::Write<o::LO> ptcl_done(numPtcls, 0);
  o::LO maxSearch = 100;
  //search all particles from this element
  auto lamb2 = OMEGA_H_LAMBDA(const int ip) {
    bool found = false;
    o::Vector<3> pos;
    o::Vector<4> bcc;
    o::LO elem = elmBeg, is=0;
    while(!found) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);
      pos[0] = data_r[ip*dof];
      pos[1] = data_r[ip*dof+1];
      pos[2] = data_r[ip*dof+2];
      p::find_barycentric_tet(M, pos, bcc);
      if(p::all_positive(bcc, 0)) {
        if(debug > 3)
          printf(" ptcl %d detected in elem %d \n", ip, elem);
        elemIdOfPtcls_w[ip] = elem;
        ptcl_done[ip] = 1;
        Kokkos::atomic_fetch_add(&numPtclsInElems_w[elem], 1);  //TODO
        found = true;
      } else {
        o::LO minInd = p::min_index(bcc, 4);
        auto dface_ind = dual_elems[elem];
        o::LO findex = 0;
        for(auto iface = elem*4; iface < (elem+1)*4; ++iface) {
          auto face_id = down_r2fs[iface];
          bool exposed = side_is_exposed[face_id];
          if(!exposed) {
            if(findex == minInd)
              elem = dual_faces[dface_ind];
            ++dface_ind;
          }
          ++findex;
        }//for
      }
      if(is > maxSearch)
        break;
      ++is;
    }
  };
  o::parallel_for(numPtcls, lamb2, "init_impurity_ptcl2");
  o::LOs ptcl_done_r(ptcl_done);
  auto minFlag = o::get_min(ptcl_done_r);
  OMEGA_H_CHECK(minFlag == 1);
  numPtclsInElems = o::LOs(numPtclsInElems_w);
  elemIdOfPtcls = o::LOs(elemIdOfPtcls_w); 
}

// using ptcl sequential numbers 0..numPtcls
void GitrmParticles::convertInitPtclElemIdsToCSR(const o::LOs& numPtclsInElems,
  o::LOs& ptclIdPtrsOfElem, o::LOs& ptclIdsInElem, o::LOs& elemIdOfPtcls,
   o::LO numPtcls) {
  o::LO verbose = 3;
  auto nel = mesh.nelems();
  auto totalPtcls = calculateCsrIndices(numPtclsInElems, ptclIdPtrsOfElem);
  OMEGA_H_CHECK(numPtcls == totalPtcls);
  // csr data
  o::Write<o::LO> ptclIdsInElem_w(numPtcls, -1);
  o::Write<o::LO> ptclsFilledInElem(nel, 0); 
  auto convert = OMEGA_H_LAMBDA(o::LO id) {
    auto el = elemIdOfPtcls[id];
    auto old = Kokkos::atomic_fetch_add(&ptclsFilledInElem[el], 1);
    OMEGA_H_CHECK(old < numPtclsInElems[el]);
    //elemId is sequential from 0 .. nel
    auto beg = ptclIdPtrsOfElem[el];
    auto pos = beg + old;
    auto elLim =  numPtclsInElems[el]; 
    if(verbose > 4)
      printf("id:el,old,beg,pos %d %d %d %d %d limit %d \n",id,el,old,beg,pos, elLim);
    OMEGA_H_CHECK(pos < ptclIdPtrsOfElem[el+1]);
    auto prev = Kokkos::atomic_exchange(&ptclIdsInElem_w[pos], id);
    if(verbose > 4)
      printf("previd %d \n", prev);
  };
  o::parallel_for(totalPtcls, convert, "Convert to CSR write");
  ptclIdsInElem = o::LOs(ptclIdsInElem_w);   
}

//To use scs, SCS_LAMBDA is required, not parallel_for(#ptcls,OMEGA_H_LAMBDA
// since ptcls in each element is iterated in groups. Construct SCS with 
// #particles in each elem passed in, otherwise newly added particles in 
// originally empty elements won't show up in SCS_LAMBDA iterations. 
// ie. their mask will be 0. If mask is not used, invalid particles may 
// show up from other threads in the launch group.
void GitrmParticles::setImpurityPtclInitData(o::LO numPtcls, const o::Reals& data, 
   const o::LOs& ptclIdPtrsOfElem, const o::LOs& ptclIdsInElem, 
   const o::LOs& elemIdOfPtcls, int maxLoops) {
  //o::LO debug =1;
  MESHDATA(mesh);

  auto size = data.size();
  auto dof = numPtcls/size;
  //TODO
  dof = 6;
  OMEGA_H_CHECK(dof==6);
  auto x_scs_d = scs->template get<PTCL_POS>();
  auto x_scs_prev_d = scs->template get<PTCL_POS_PREV>();
  auto vel_d = scs->template get<PTCL_VEL>();
  auto fid_d = scs->template get<XPOINT_FACE>();
  auto pid_scs = scs->template get<PTCL_ID>();
  o::Write<o::LO> nextPtclInd(mesh.nelems(), 0);
  auto lambda = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);
      o::Vector<4> bcc;
      auto pos = o::zero_vector<3>();
      o::Real vel[] = {0,0,0};
      auto nextInd = Kokkos::atomic_fetch_add(&nextPtclInd[elem], 1);
      auto ind = ptclIdPtrsOfElem[elem] + nextInd;
      //TODO check =
      OMEGA_H_CHECK(ind < ptclIdPtrsOfElem[elem+1]);
      auto ip = ptclIdsInElem[ind]; //ip 0..numPtcls
      OMEGA_H_CHECK(elemIdOfPtcls[ip] == elem);

      pos[0] = data[ip*dof];
      pos[1] = data[ip*dof+1];
      pos[2] = data[ip*dof+2];
      vel[0] = data[ip*dof+3];
      vel[1] = data[ip*dof+4];
      vel[2] = data[ip*dof+5];                                     
      p::find_barycentric_tet(M, pos, bcc);
      OMEGA_H_CHECK(p::all_positive(bcc, 0));
      for(int i=0; i<3; i++) {
        x_scs_prev_d(pid,i) = pos[i];
        x_scs_d(pid,i) = pos[i];
        vel_d(pid, i) = vel[i];
      }

      fid_d(pid) = -1;
      pid_scs(pid) = pid;
    }
  };
  scs->parallel_for(lambda);
}

//Depends on netcdf format, and semicolon at the end of fields
void GitrmParticles::processPtclInitFile(const std::string &fName,
    o::HostWrite<o::Real> &data, PtclInitStruct &ps, o::LO numPtcls) {
  o::LO verbose = 1;
  std::ifstream ifs(fName);
  if (!ifs.is_open()) {
    std::cout << "Error opening PtclInitFile file " << fName << '\n';
    //exit(1); //TODO
  }
  // can't set in ps, since field names in ps used below not from array
  constexpr int nComp = 6;

  OMEGA_H_CHECK(ps.nComp == nComp);
  bool foundNP, dataInit, foundComp[nComp], dataLine[nComp]; //6=x,y,z,vx,vy,vz
  std::string fieldNames[nComp];
  int ind[nComp];
  for(int i = 0; i < nComp; ++i) {
    ind[i] = 0;
    foundComp[i] = dataLine[i] = false;
  }

  fieldNames[0] = ps.xName;
  fieldNames[1] = ps.yName;
  fieldNames[2] = ps.zName;
  fieldNames[3] = ps.vxName;
  fieldNames[4] = ps.vyName;
  fieldNames[5] = ps.vzName;      
  foundNP = dataInit = false;
  std::string line, s1, s2, s3;
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
    if(verbose >4){
          std::cout << "str s1:" << s1 << "\n";
    }
    // Skip blank line
    if(s1.find_first_not_of(' ') == std::string::npos) {
      s1 = "";
      if(!semi)
       continue;
    }
    if(s1 == ps.nPname) {
      ss >> s2 >> s3;
      OMEGA_H_CHECK(s2 == "=");
      ps.nP = std::stoi(s3);
      if(numPtcls <= 0)
        numPtcls = ps.nP;
      else if(numPtcls < ps.nP)
        ps.nP = numPtcls;
      foundNP = true;
      if(verbose >4)
          std::cout << "nP:" << ps.nP << " Using numPtcls " << numPtcls << "\n";
    }
    if(!dataInit && foundNP) {
      data = o::HostWrite<o::Real>(ps.nComp*ps.nP); //destruct ?
      dataInit = true;
    }
    int compBeg = 0, compEnd = nComp;
    // if ; ends data of each parameters, otherwise comment this block
    // to search for each parameter for every data line
    for(int iComp = 0; iComp<nComp; ++iComp) {
      if(dataInit && dataLine[iComp]) {
        compBeg = iComp;
        compEnd = compBeg + 1;
      }
    }
   
    if(dataInit) {
      for(int iComp = compBeg; iComp<compEnd; ++iComp) {
        parseFileFieldData(ss, s1, fieldNames[iComp], semi, data, ind[iComp], 
          dataLine[iComp], iComp, nComp);
        
        if(!foundComp[iComp] && dataLine[iComp])
          foundComp[iComp] = true;
      }
    }
    s1 = s2 = s3 = "";
  } //while

  OMEGA_H_CHECK(dataInit && foundNP);
  for(int i=0; i<6; ++i)
    OMEGA_H_CHECK(foundComp[i]==true);

  if(ifs.is_open()) {
    ifs.close();
  }
}

void GitrmParticles::setImpurityPtclInitRndDistribution(
    o::Write<o::LO> &elemAndFace) {
  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto down_r2fs = down_r2f.ab2b;
  const auto dual_faces = dual.ab2b;
  const auto dual_elems = dual.a2ab;

  //Set particle coordinates. Initialized only on one face. TODO confirm this ? 
  auto x_scs_d = scs->template get<PTCL_POS>();
  auto x_scs_prev_d = scs->template get<PTCL_POS_PREV>();
  auto vel_d = scs->template get<PTCL_VEL>();
  auto fid_d = scs->template get<XPOINT_FACE>();
  
  // TODO replace this with device rnd; now same number goes to all particles
  std::srand(time(NULL));
  auto rnd1 = (double)(std::rand())/RAND_MAX;
  auto rnd2 = (double)(std::rand())/RAND_MAX;
  auto rnd3 = (double)(std::rand())/RAND_MAX;
  auto rnd = (double)(std::rand())/RAND_MAX;

  o::Write<o::LO> elem_ids(scs->capacity(),-1);
  auto lambda = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
    //if(elemAndFace[1] >=0 && elem == elemAndFace[1]) {  
      o::LO verbose =1;
      // TODO if more than an element  ?
      const auto faceId = elemAndFace[2];
      const auto fv2v = o::gather_verts<3>(face_verts, faceId);
      const auto face = p::gatherVectors3x3(coords, fv2v);
      auto fcent = p::find_face_centroid(faceId, coords, face_verts);
      auto tcent = p::centroid_of_tet(elem, mesh2verts, coords); 
      auto diff = tcent - fcent;
      if(verbose >3)
        printf(" elemAndFace[1]:%d, elem:%d face%d beg%d\n", 
          elemAndFace[1], elem, elemAndFace[2], elemAndFace[0]);

      o::Vector<4> bcc;
      o::Vector<3> pos;
      do {   
        o::Real bc1 = (rnd1 > rnd2)?rnd2:rnd1;
        o::Real bc2 = std::abs(rnd1 - rnd2);
        o::Real bc3 = 1.0 - bc1 - bc2;
        o::Vector<3> fpos = bc1*face[0] + bc2*face[1] + bc3*face[2];
        auto fnorm = p::find_face_normal(faceId, elem, coords, mesh2verts, 
                                        face_verts, down_r2fs);
        pos = fpos - 0.000001*fnorm;
        auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
        auto M = p::gatherVectors4x3(coords, tetv2v);
        p::find_barycentric_tet(M, pos, bcc);

      } while(!p::all_positive(bcc, 0));

      double amu = 2.0; //TODO
      double energy[] = {4.0, 4, 4}; //TODO actual [4,0,0]
      double vel[] = {0,0,0};
      for(int i=0; i<3; i++) {
        x_scs_prev_d(pid,i) = pos[i];
        x_scs_d(pid,i) = pos[i];
        auto en = energy[i];   
        //if(! p::almost_equal(energy[i], 0))
        vel[i] = std::sqrt(2.0 * abs(en) * 1.60217662e-19 / (amu * 1.6737236e-27));
        vel[i] *= rnd;  //TODO
      }

      for(int i=0; i<3; i++)
        vel_d(pid, i) = vel[i];

      fid_d(pid) = -1;
 
      elem_ids[pid] = elem;

      if(verbose >2)
        printf("elm %d : pos %.4f %.4f %.4f : vel %.1f %.1f %.1f Mask%d\n",
          elem, x_scs_prev_d(pid,0), x_scs_prev_d(pid,1), x_scs_prev_d(pid,2),
          vel[0], vel[1], vel[2], mask);
    }
  };
  scs->parallel_for(lambda);
}

// spherical coordinates (wikipedia), radius r=1.5m, inclination theta[0,pi] from the z dir,
// azimuth angle phi[0, 2π) from the Cartesian x-axis (so that the y-axis has phi = +90°).
void GitrmParticles::findInitialBdryElemIdInADir(o::Real theta, o::Real phi, o::Real r,
     o::LO &initEl, o::Write<o::LO> &elemAndFace, o::LO maxLoops, o::Real outer){

  o::LO debug = 4;

  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);

  const auto down_r2fs = down_r2f.ab2b;
  const auto dual_faces = dual.ab2b;
  const auto dual_elems = dual.a2ab;

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
      auto dface_ind = dual_elems[elem];
      const auto beg_face = elem *4;
      const auto end_face = beg_face +4;
      o::LO fIndex = 0;

      for(auto iface = beg_face; iface < end_face; ++iface) {
        const auto face_id = down_r2fs[iface];

        o::Vector<3> xpoint = o::zero_vector<3>();
        const auto face = p::get_face_of_tet(mesh2verts, coords, elem, fIndex);
        bool detected = p::line_triangle_intx_simple(face, orig, dest, xpoint);
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
          auto adj_elem  = dual_faces[dface_ind];
          elem = adj_elem;
          if(debug >4) {
            printf(" faceid %d detected on interior; next elm %d\n", face_id, elem);
          }
          break;
        }
        if(!side_is_exposed[face_id]){
          ++dface_ind;
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
  OMEGA_H_CHECK(initEl>=0 && elemId_fh[0]>=0);

}
