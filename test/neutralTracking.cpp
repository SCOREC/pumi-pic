#include <vector>
#include <fstream>
#include <iostream>
#include "pumipic_adjacency.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_file.hpp"
#include "GitrmParticles.hpp"

namespace o = Omega_h;
namespace p = pumipic;

void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

void markDetectorCylinder(o::Mesh& mesh, bool renderPiscesCylCells) {
  o::HostWrite<o::LO> fIds_h{277, 609, 595, 581, 567, 553, 539, 
    525, 511, 497, 483, 469, 455, 154};
  o::LOs faceIds(fIds_h);
  auto numFaceIds = faceIds.size();
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
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
          auto elmId = p::elem_of_bdry_face(i, f2r_ptr, f2r_elem);
          elemTagIds[elmId] = id;
        }
      }
    }
  });

  mesh.add_tag<o::LO>(o::FACE, "piscesTiRod_ind", 1, o::LOs(faceTagIds));
  mesh.add_tag<o::LO>(o::REGION, "piscesTiRodId", 1, o::LOs(elemTagIds));
}

void updatePtclPositions(SCS* scs) {
  auto x_scs_d = scs->get<0>();
  auto xtgt_scs_d = scs->get<1>();
  auto updatePtclPos = SCS_LAMBDA(const int&, const int& pid, const bool&) {
    x_scs_d(pid,0) = xtgt_scs_d(pid,0);
    x_scs_d(pid,1) = xtgt_scs_d(pid,1);
    x_scs_d(pid,2) = xtgt_scs_d(pid,2);
    xtgt_scs_d(pid,0) = 0;
    xtgt_scs_d(pid,1) = 0;
    xtgt_scs_d(pid,2) = 0;
  };
  scs->parallel_for(updatePtclPos, "updatePtclPos");
}

void rebuild(SCS* scs, o::LOs elem_ids) {
  //fprintf(stderr, "rebuilding..\n");
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto pid_d =  scs->get<2>();
  SCS::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void)e;
    scs_elem_ids(pid) = elem_ids[pid];
  };
  scs->parallel_for(lamb, "rebuild");
  scs->rebuild(scs_elem_ids);
}

void storePiscesData(SCS* scs, o::Mesh& mesh, o::Write<o::LO>& data_d, 
  o::Write<o::Real>& xpoints, o::Write<o::LO>& xface_ids, o::LO iter, 
  bool debug=true) {
  double radMax = 0.05; //m 0.0446+0.005
  double zMin = 0; //m height min
  double zMax = 0.15; //m height max 0.14275
  double htBead1 =  0.01275; //m ht of 1st bead
  double dz = 0.01; //m ht of beads 2..14
  auto pisces_ids = mesh.get_array<o::LO>(o::FACE, "piscesTiRod_ind");
  auto pid_scs = scs->get<PTCL_ID>();

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto fid = xface_ids[pid];
    if(mask >0 && fid>=0) {
      // test
      o::Vector<3> xpt;
      for(o::LO i=0; i<3; ++i)
        xpt[i] = xpoints[pid*3+i];
      auto x = xpt[0], y = xpt[1], z = xpt[2];
      o::Real rad = std::sqrt(x*x + y*y);
      o::LO zInd = -1;
      if(rad < radMax && z <= zMax && z >= zMin)
        zInd = (z > htBead1) ? (1+(o::LO)((z-htBead1)/dz)) : 0;
      
      auto detId = pisces_ids[fid];
      if(detId >=0) {
        //if(debug)
          printf("ptclID %d zInd %d detId %d pos %.5f %.5f %.5f iter %d\n", 
            pid_scs(pid), zInd, detId, x, y, z, iter);
        Kokkos::atomic_fetch_add(&data_d[detId], 1);
      }
    }
  };
  scs->parallel_for(lamb, "storePiscesData");
}

void neutralBorisMove(SCS* scs,  const o::Real dTime) {
  auto vel_scs = scs->get<PTCL_VEL>();
  auto tgt_scs = scs->get<PTCL_NEXT_POS>();
  auto pos_scs = scs->get<PTCL_POS>();
  auto boris = SCS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto vel = p::makeVector3(pid, vel_scs);
      auto pos = p::makeVector3(pid, pos_scs);
      // Next position and velocity
      tgt_scs(pid, 0) = pos[0] + vel[0] * dTime;
      tgt_scs(pid, 1) = pos[1] + vel[1] * dTime;
      tgt_scs(pid, 2) = pos[2] + vel[2] * dTime;
      vel_scs(pid, 0) = vel[0];
      vel_scs(pid, 1) = vel[1];
      vel_scs(pid, 2) = vel[2];      
    }// mask
  };
  scs->parallel_for(boris, "neutralBorisMove");
} 

void neutralBorisMove_float(SCS* scs,  const o::Real dTime, bool debug = false) {
  auto vel_scs = scs->get<PTCL_VEL>();
  auto tgt_scs = scs->get<PTCL_NEXT_POS>();
  auto pos_scs = scs->get<PTCL_POS>();
  auto boris = SCS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto vel = p::makeVector3(pid, vel_scs);
      auto pos = p::makeVector3(pid, pos_scs);
      // Next position and velocity
      float val[3], v2[3];
      for(int i=0; i<3; ++i) {
        val[i] = float(pos[i]) + float(vel[i]) * float(dTime);
        v2[i] = float(vel[i]);
        if(debug)
          printf(" %f", val[i]);
      }
      if(debug)
        printf("\n");

      for(int i=0; i<3; ++i) {      
        tgt_scs(pid, i) = val[i];
        vel_scs(pid, i) = v2[i];
      }
    
    }// mask
  };
  scs->parallel_for(boris, "neutralBorisMove");
} 

void printPtclHostData(o::HostWrite<o::Real>& dh, int num, int dof,
  const char* name, int iter=-1) { 
  double pos[3], vel[3];
  for(int i=0; i<num; ++i) {
    for(int j=0; j<3; ++j) {
      pos[j] = dh[i*dof+j];
      vel[j] = dh[i*dof+3+j];
    }
    int id = static_cast<int>(dh[i*dof + 6]);
    int it = static_cast<int>(dh[i*dof + 7]);
    printf("%s %d pos %g %g %g iter %d vel %g %g %g updateiter %d\n",
      name, id+1, pos[0], pos[1], pos[2], iter, vel[0], vel[1], vel[2], it);
  }
} 


void updateStepData(SCS* scs, int iter, int numPtcls, 
  o::Write<o::Real>& ptclsDataAll, o::Write<o::Real>& data, int dof=8) {
  // size is not scs->nPtcls, since id: 0 to numPtcls
  auto vel_scs = scs->get<PTCL_VEL>();
  auto pos_scs = scs->get<PTCL_POS>();
  auto pid_scs = scs->get<PTCL_ID>();
  auto step = SCS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto id = pid_scs(pid);
      auto vel = p::makeVector3(pid, vel_scs);
      auto pos = p::makeVector3(pid, pos_scs);
      // NOTE: particle id from 0 to numPtcls; only if indices are used as ids.
      // If pid is used as id's, then range will be > numPtcls => out of memory. 
      for(int i=0; i<3; ++i) {
        data[id*dof+i] = pos[i];
        data[id*dof+3+i] = vel[i];
      }
      data[id*dof + 6] = id;
      data[id*dof + 7] = iter;
      
      for(int i=0; i<8; ++i)
        ptclsDataAll[id*dof+i] = data[id*dof+i];

      //NOTE: many of these prints go missing
      //printf("ptclHistory %d zInd -1 detId -1 pos %g %g %g iter %d vel %g %g %g\n",
      // id+1, pos[0], pos[1], pos[2], iter, vel[0], vel[1], vel[2]);
    }// mask
  };
  scs->parallel_for(step, "updateStepData");
}

void printStepData(SCS* scs, int iter, int numPtcls, o::Write<o::Real>& ptclsDataAll,
   o::Write<o::Real>& data, int dof=8, bool accum = false) {

  printf("total_ptcls %d\n", scs->nPtcls());
  if(iter ==0) {
    updateStepData(scs, iter, numPtcls, ptclsDataAll, data, dof);
  }

  if(accum) {
    o::HostWrite<o::Real> ptclAllHost(ptclsDataAll);
    printPtclHostData(ptclAllHost, numPtcls, dof, "ptclHistory_all", iter);
  } else {
    o::HostWrite<o::Real> dh(data);
    o::HostWrite<o::Real> dp(scs->nPtcls() * dof);
    for(int ip=0, n=0; ip< numPtcls; ++ip) {
      //only ptcls from scs->nptcls have valid ids
      int id = static_cast<int>(dh[ip*dof+6]);
      if(id < 0)
        continue;
      for(int i=0; i<8; ++i) {
        dp[n*dof+i] = dh[id*dof+i];
      }
      ++n;
    }
    printPtclHostData(dp, scs->nPtcls(), dof, "ptclHistory", iter);
  }
}

void search(SCS* scs, o::Mesh& mesh, int iter, o::Write<o::LO>& data_d, 
  o::Write<o::Real>& xpoints_d, bool debug=false ) {
  
  Kokkos::Profiling::pushRegion("gitrm_search");
  assert(scs->nElems() == mesh.nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  //o::Write<o::Real>xpoints_d(3*scsCapacity, 0, "xpoints");
  o::Write<o::LO>xface_ids(scsCapacity, -1, "xface_ids");
  //o::fill(xface_ids, -1);

  auto x_scs = scs->get<0>();
  auto xtgt_scs = scs->get<1>();
  auto pid_scs = scs->get<2>();
  bool isFound = p::search_mesh<Particle>(mesh, scs, x_scs, xtgt_scs, pid_scs, 
    elem_ids, xpoints_d, xface_ids, maxLoops);
  assert(isFound);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("storePiscesData");
  //auto elm_ids = o::LOs(elem_ids);
  //o::Reals collisionPoints = o::Reals(xpoints_d);
  //o::LOs collisionPointFaceIds = o::LOs(xface_ids);
 
  //output particle positions, for converting to vtk
  storePiscesData(scs, mesh, data_d, xpoints_d, xface_ids, iter, debug);
  Kokkos::Profiling::popRegion();
  //update positions and set the new element-to-particle lists
  Kokkos::Profiling::pushRegion("rebuild");
  rebuild(scs, elem_ids);
  Kokkos::Profiling::popRegion();
}

int main(int argc, char** argv) {
  auto start_sim = std::chrono::system_clock::now(); 
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  printf("particle_structs floating point value size (bits): %zu\n", sizeof(fp_t));
  printf("omega_h floating point value size (bits): %zu\n", sizeof(Omega_h::Real));
  printf("Kokkos execution space memory %s name %s\n",
      typeid (Kokkos::DefaultExecutionSpace::memory_space).name(),
      typeid (Kokkos::DefaultExecutionSpace).name());
  printf("Kokkos host execution space %s name %s\n",
      typeid (Kokkos::DefaultHostExecutionSpace::memory_space).name(),
      typeid (Kokkos::DefaultHostExecutionSpace).name());
  printTimerResolution();
  if(argc < 2)
  {
    std::cout << "Usage: " << argv[0] 
      << " <mesh><ptcls_file>[<nPtcls><nIter> <histInterval> <timeStep>]\n";
    exit(1);
  }
 
  auto mesh = Omega_h::read_mesh_file(argv[1], lib.self());
  printf("Number of elements %d verts %d\n", mesh.nelems(), mesh.nverts());
  std::string ptclSource = argv[2];
  bool piscesRun = true;
  bool debug = false;
  bool chargedTracking = false; //false for neutral tracking

  if(!chargedTracking)
    printf("WARNING: neutral particle tracking is ON \n");

  if(piscesRun)
    markDetectorCylinder(mesh, true);

  int numPtcls = 100000;
  double dTime = 5e-9; //pisces:5e-9 for 100,000 iterations
  int NUM_ITERATIONS = 100000; //higher beads needs >10K
  int histInterval = 0;
  if(argc > 3)
    numPtcls = atoi(argv[3]);
  if(argc > 4)
    NUM_ITERATIONS = atoi(argv[4]);
  if(argc > 5)
    histInterval = atoi(argv[5]);
  if(argc > 6)
    dTime = atof(argv[6]);

  GitrmParticles gp(mesh, dTime);
  //current extruded mesh has Y, Z switched
  // ramp: 330, 90, 1.5, 200,10; tgt 324, 90...; upper: 110, 0
  printf("Initializing Particles\n");

  gp.initImpurityPtclsFromFile(ptclSource, numPtcls, 100, false);

  auto &scs = gp.scs;
  //what if scs capacity increases ?
  const auto scsCapacity = scs->capacity();
  o::Write<o::Real>xpoints_d(4*scsCapacity, 0, "xpoints");
  //o::Write<o::LO>xface_ids((int)(1.5*scsCapacity), -1, "xface_ids"); //crash

  o::LO numGrid = 14;
  o::Write<o::LO>data_d(numGrid, 0);

  printf("\ndTime %g NUM_ITERATIONS %d\n", dTime, NUM_ITERATIONS);
  int dofStepData = 8;
  o::Write<o::Real> ptclsDataAll(numPtcls*dofStepData);

  fprintf(stderr, "\n*********Main Loop**********\n");
  auto end_init = std::chrono::system_clock::now();
  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    if(scs->nPtcls() == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      fprintf(stderr, "Total iterations = %d\n", iter);
      break;
    }
    if(iter%100 ==0)
      fprintf(stderr, "=================iter %d===============\n", iter);

    o::Write<o::Real> data(numPtcls*dofStepData, -1);
    if(iter==0 && histInterval >0)
      printStepData(scs, 0, numPtcls, ptclsDataAll, data, dofStepData, true);

    //neutralBorisMove(scs, dTime);
    neutralBorisMove_float(scs, dTime);

    search(scs, mesh, iter, data_d, xpoints_d, debug);

    if(histInterval >0) {
      updateStepData(scs, iter+1, numPtcls, ptclsDataAll, data, dofStepData); 
      if((iter+1)%histInterval == 0)
      printStepData(scs, iter+1, numPtcls, ptclsDataAll, data, dofStepData, true); //accum
    }
    if(iter%100 ==0)
      fprintf(stderr, "nPtcls %d\n", scs->nPtcls());
  }
  auto end_sim = std::chrono::system_clock::now();
  std::chrono::duration<double> dur_init = end_init - start_sim;
  std::cout << "Initialization duration " << dur_init.count()/60 << " min.\n";
  std::chrono::duration<double> dur_steps = end_sim - end_init;
  std::cout << "Total Main Loop duration " << dur_steps.count()/60 << " min.\n";
  if(piscesRun) {
    std::cout << "Pisces detections \n";
    printGridData(data_d);
  }
  fprintf(stderr, "done\n");
  return 0;
}


