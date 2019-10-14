#include <vector>
#include <fstream>
#include <iostream>
#include "Omega_h_file.hpp"
#include <Kokkos_Core.hpp>
#include <Omega_h_mesh.hpp>

#include "pumipic_mesh.hpp"
#include "pumipic_adjacency.hpp"

#include "GitrmParticles.hpp"
#include "GitrmPush.hpp"

#define HISTORY 0


void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
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

void storePiscesDataSeparate(SCS* scs, o::Mesh* mesh, o::Write<o::LO>& data_d, 
  o::Write<o::Real>& xpoints, o::Write<o::LO>& xface_ids, o::LO iter, 
  bool debug=true) {
  double radMax = 0.05; //m 0.0446+0.005
  double zMin = 0; //m height min
  double zMax = 0.15; //m height max 0.14275
  double htBead1 =  0.01275; //m ht of 1st bead
  double dz = 0.01; //m ht of beads 2..14
  auto pisces_ids = mesh->get_array<o::LO>(o::FACE, "piscesTiRod_ind");
  auto pid_scs = scs->get<PTCL_ID>();

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto fid = xface_ids[pid];
    if(mask >0 && fid>=0) {
      // test
      o::Vector<3> xpt;
      for(o::LO i=0; i<3; ++i)
        xpt[i] = xpoints[pid*3+i];
      auto x = xpt[0], y = xpt[1], z = xpt[2];
      o::Real rad = sqrt(x*x + y*y);
      o::LO zInd = -1;
      if(rad < radMax && z <= zMax && z >= zMin)
        zInd = (z > htBead1) ? (1+(o::LO)((z-htBead1)/dz)) : 0;
      
      auto detId = pisces_ids[fid];
      if(detId >=0) {
        if(debug)
          printf("ptclID %d zInd %d detId %d pos %.5f %.5f %.5f iter %d\n", 
            pid_scs(pid), zInd, detId, x, y, z, iter);
        Kokkos::atomic_fetch_add(&(data_d[detId]), 1);
      }
    }
  };
  scs->parallel_for(lamb, "storePiscesData");
}


void rebuild(p::Mesh& picparts, SCS* scs, o::LOs elem_ids, const bool output) {
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto ids = scs->get<2>();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(output && mask > 0)
      printf("elem_ids[%d] %d ptcl_id:%d\n", pid, elem_ids[pid], ids(pid));
  };
  scs->parallel_for(printElmIds);

  SCS::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);
  SCS::kkLidView scs_process_ids("scs_process_ids", scs_capacity);
  Omega_h::LOs is_safe = picparts.safeTag();
  Omega_h::LOs elm_owners = picparts.entOwners(picparts.dim());
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      int new_elem = elem_ids[pid];
      scs_elem_ids(pid) = new_elem;
      scs_process_ids(pid) = comm_rank;
      if (new_elem != -1 && is_safe[new_elem] == 0) {
        scs_process_ids(pid) = elm_owners[new_elem];
      }
    }
  };
  scs->parallel_for(lamb);
  
  scs->migrate(scs_elem_ids, scs_process_ids);

  ids = scs->get<2>();
  if (output) {
    auto printElms = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0)
        printf("Rank %d Ptcl: %d has Element %d and id %d\n", comm_rank, pid, e, ids(pid));
    };
    scs->parallel_for(printElms);
  }
}

void search(p::Mesh& picparts, SCS* scs, GitrmParticles& gp, int iter, 
  o::Write<o::LO>& data_d, o::Write<o::Real>& xpoints_d, bool debug=false ) {
  o::Mesh* mesh = picparts.mesh();
  Kokkos::Profiling::pushRegion("gitrm_search");
  if(debug)
    printf("elems scs %d mesh %d\n", scs->nElems(), mesh->nelems());
  assert(scs->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 10;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  o::Write<o::LO>xface_ids(scsCapacity, -1, "xface_ids");
  auto x_scs = scs->get<0>();
  auto xtgt_scs = scs->get<1>();
  auto pid_scs = scs->get<2>();

  bool isFound = p::search_mesh_3d<Particle>(*mesh, scs, x_scs, xtgt_scs, pid_scs, 
    elem_ids, xpoints_d, xface_ids, maxLoops, debug);
  assert(isFound);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("storePiscesData");

  storePiscesDataSeparate(scs, mesh, data_d, xpoints_d, xface_ids, iter, debug);
  //storePiscesData(gp, data_d, iter, debug);
  Kokkos::Profiling::popRegion();
  //update positions and set the new element-to-particle lists
  Kokkos::Profiling::pushRegion("rebuild");
  rebuild(picparts, scs, elem_ids, debug);
  Kokkos::Profiling::popRegion();
}

o::Mesh readMesh(char* meshFile, o::Library& lib) {
  const auto rank = lib.world()->rank();
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    if(!rank)
      std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    if(!rank)
      std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self(), true);
  } else {
    if(!rank)
      std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}


int main(int argc, char** argv) {
  auto start_sim = std::chrono::system_clock::now(); 
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if(argc < 2)
  {
    if(comm_rank == 0)
      std::cout << "Usage: " << argv[0] 
        << " <mesh><ptcls_file>[<nPtcls><nIter> <histInterval> <timeStep>]\n";
    exit(1);
  }
  auto deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  //TODO 
  assert(deviceCount==1);
  assert(comm_size==1);

  if(comm_rank == 0) {
    printf("device count per process %d\n", deviceCount);
    printf("world ranks %d\n", comm_size);
    printf("particle_structs floating point value size (bits): %zu\n", sizeof(fp_t));
    printf("omega_h floating point value size (bits): %zu\n", sizeof(Omega_h::Real));
    printf("Kokkos execution space memory %s name %s\n",
        typeid (Kokkos::DefaultExecutionSpace::memory_space).name(),
        typeid (Kokkos::DefaultExecutionSpace).name());
    printf("Kokkos host execution space %s name %s\n",
        typeid (Kokkos::DefaultHostExecutionSpace::memory_space).name(),
        typeid (Kokkos::DefaultHostExecutionSpace).name());
    printTimerResolution();
  }
  auto full_mesh = readMesh(argv[1], lib);
  MPI_Barrier(MPI_COMM_WORLD);

  Omega_h::HostWrite<Omega_h::LO> host_owners(full_mesh.nelems());
  if (comm_size > 1) {
    std::ifstream in_str(argv[2]); //TODO update 
    if (!in_str) {
      if (!comm_rank)
        fprintf(stderr,"Cannot open file %s\n", argv[2]);
      return EXIT_FAILURE;
    }
    int own;
    int index = 0;
    while(in_str >> own)
      host_owners[index++] = own;
  }
  else
    for (int i = 0; i < full_mesh.nelems(); ++i)
      host_owners[i] = 0;
  Omega_h::Write<Omega_h::LO> owner(host_owners);

  //Create Picparts with the full mesh
  p::Mesh picparts(full_mesh, owner);
  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts(); //caching adjacency info
 
  if (comm_rank == 0)
    printf("Mesh loaded with verts %d edges %d faces %d elements %d\n", 
      mesh->nverts(), mesh->nedges(), mesh->nfaces(), mesh->nelems());
  //TODO FIXME argv[2] is used in mutliple places !
  std::string ptclSource = argv[2];
  bool piscesRun = true;
  bool chargedTracking = false; //false for neutral tracking

  if(!chargedTracking)
    printf("WARNING: neutral particle tracking is ON \n");

  int numPtcls = 100000;
  double dTime = 5e-9; //pisces:5e-9 niter 100,000
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

  std::ofstream ofsHistory;
  if(histInterval > 0)
    ofsHistory.open("history.txt");
  

  bool debug = false;
  
  GitrmParticles gp(*mesh, dTime);
  GitrmMesh gm(*mesh);
  if(piscesRun)
    gm.markPiscesCylinder(true);
  //current extruded mesh has Y, Z switched
  // ramp: 330, 90, 1.5, 200,10; tgt 324, 90...; upper: 110, 0
  if(debug)
    printf("Initializing Particles\n");
  gp.initPtclsFromFile(picparts, ptclSource, numPtcls, 100, false);
  auto* scs = gp.scs;
  const auto scsCapacity = scs->capacity();
  //TODO fix this extra storage 
  o::Write<o::Real>xpoints_d(5*scsCapacity, 0, "xpoints");
  //o::Write<o::LO>xface_ids((int)(1.5*scsCapacity), -1, "xface_ids"); //crash

  o::LO numGrid = 14;
  o::Write<o::LO>data_d(numGrid, 0);

  printf("\ndTime %g NUM_ITERATIONS %d\n", dTime, NUM_ITERATIONS);
  int dofStepData = 8;
  o::Write<o::Real> ptclsDataAll(numPtcls*dofStepData);

  fprintf(stderr, "\n*********Main Loop**********\n");
  auto end_init = std::chrono::system_clock::now();
  int iter;
  int np;
  int scs_np;
  for(iter=0; iter<NUM_ITERATIONS; iter++) {
    scs_np = scs->nPtcls();
    MPI_Allreduce(&scs_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if(comm_rank == 0 && (debug || iter%1000 ==0))
      fprintf(stderr, "=================iter %d===============\n", iter);
   //TODO not ready for MPI
    #if HISTORY > 0
    o::Write<o::Real> data(numPtcls*dofStepData, -1);
    if(iter==0 && histInterval >0)
      printStepData(ofsHistory, scs, 0, numPtcls, ptclsDataAll, data, dofStepData, true);
    #endif

    Kokkos::Profiling::pushRegion("neutralBorisMove");
    //neutralBorisMove(scs, dTime);
    neutralBorisMove_float(scs, dTime);
    Kokkos::Profiling::popRegion();

    MPI_Barrier(MPI_COMM_WORLD);
    
    search(picparts, scs, gp, iter, data_d, xpoints_d, debug);
    #if HISTORY > 0
    if(histInterval >0) {
      updateStepData(scs, iter+1, numPtcls, ptclsDataAll, data, dofStepData); 
      if((iter+1)%histInterval == 0)
        printStepData(ofsHistory, scs, iter+1, numPtcls, ptclsDataAll, data, 
        dofStepData, true); //last accum
    }
    #endif
    if(comm_rank == 0 && iter%1000 ==0)
      fprintf(stderr, "nPtcls %d\n", scs->nPtcls());
    scs_np = scs->nPtcls();
    MPI_Allreduce(&scs_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
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


