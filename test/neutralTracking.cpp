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


void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

void updatePtclPositions(PS* ptcls) {
  auto x_ps_d = ptcls->get<0>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto updatePtclPos = PS_LAMBDA(const int&, const int& pid, const bool&) {
    x_ps_d(pid,0) = xtgt_ps_d(pid,0);
    x_ps_d(pid,1) = xtgt_ps_d(pid,1);
    x_ps_d(pid,2) = xtgt_ps_d(pid,2);
    xtgt_ps_d(pid,0) = 0;
    xtgt_ps_d(pid,1) = 0;
    xtgt_ps_d(pid,2) = 0;
  };
  ps::parallel_for(ptcls, updatePtclPos, "updatePtclPos");
}

void updateSurfaceDetectionSeparate(PS* ptcls, o::Mesh* mesh, o::Write<o::LO>& data_d, 
  o::Write<o::Real>& xpoints, o::Write<o::LO>& xface_ids, o::LO iter, 
  bool debug=true) {
  double radMax = 0.05; //m 0.0446+0.005
  double zMin = 0; //m height min
  double zMax = 0.15; //m height max 0.14275
  double htBead1 =  0.01275; //m ht of 1st bead
  double dz = 0.01; //m ht of beads 2..14
  auto pisces_ids = mesh->get_array<o::LO>(o::FACE, "DetectorSurfaceIndex");
  auto pid_ps = ptcls->get<PTCL_ID>();

  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto fid = xface_ids[ptcl];
      if(fid>=0) {
        xface_ids[ptcl] = -1;
        // test
        o::Vector<3> xpt;
        for(o::LO i=0; i<3; ++i)
          xpt[i] = xpoints[ptcl*3+i];
        auto x = xpt[0], y = xpt[1], z = xpt[2];
        o::Real rad = sqrt(x*x + y*y);
        o::LO zInd = -1;
        if(rad < radMax && z <= zMax && z >= zMin)
          zInd = (z > htBead1) ? (1+(o::LO)((z-htBead1)/dz)) : 0;
        
        auto detId = pisces_ids[fid];
        if(detId >=0) {
          if(debug)
            printf("ptclID %d zInd %d detId %d pos %.5f %.5f %.5f iter %d\n", 
              pid_ps(pid), zInd, detId, x, y, z, iter);
          Kokkos::atomic_increment(&(data_d[detId]));
        }
      }
    }
  };
  ps::parallel_for(ptcls, lamb, "updateSurfaceDetection");
}


void rebuild(p::Mesh& picparts, PS* ptcls, o::LOs elem_ids, const bool output) {
  updatePtclPositions(ptcls);
  const int ps_capacity = ptcls->capacity();
  auto ids = ptcls->get<2>();
  auto printElmIds = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(output && mask > 0)
      printf("elem_ids[%d] %d ptcl_id:%d\n", pid, elem_ids[pid], ids(pid));
  };
  ps::parallel_for(ptcls, printElmIds);

  PS::kkLidView ps_elem_ids("ps_elem_ids", ps_capacity);
  PS::kkLidView ps_process_ids("ps_process_ids", ps_capacity);
  Omega_h::LOs is_safe = picparts.safeTag();
  Omega_h::LOs elm_owners = picparts.entOwners(picparts.dim());
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      int new_elem = elem_ids[pid];
      ps_elem_ids(pid) = new_elem;
      ps_process_ids(pid) = comm_rank;
      if (new_elem != -1 && is_safe[new_elem] == 0) {
        ps_process_ids(pid) = elm_owners[new_elem];
      }
    }
  };
  ps::parallel_for(ptcls, lamb);
  
  ptcls->migrate(ps_elem_ids, ps_process_ids);

  ids = ptcls->get<2>();
  if (output) {
    auto printElms = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0)
        printf("Rank %d Ptcl: %d has Element %d and id %d\n", comm_rank, pid, e, ids(pid));
    };
    ps::parallel_for(ptcls, printElms);
  }
}

void search(p::Mesh& picparts, PS* ptcls, GitrmParticles& gp, int iter, 
  o::Write<o::LO>& data_d, o::Write<o::Real>& xpoints_d, 
  o::Write<o::LO>&xface_ids, bool debug=false ) {
  o::Mesh* mesh = picparts.mesh();
  Kokkos::Profiling::pushRegion("gitrm_search");
  if(debug)
    printf("elems ps %d mesh %d\n", ptcls->nElems(), mesh->nelems());
  assert(ptcls->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 20;
  const auto psCapacity = ptcls->capacity();
  assert(psCapacity >0);
  o::Write<o::LO> elem_ids(psCapacity,-1);
  auto x_ps = ptcls->get<0>();
  auto xtgt_ps = ptcls->get<1>();
  auto pid_ps = ptcls->get<2>();

  bool isFound = p::search_mesh<Particle>(*mesh, ptcls, x_ps, xtgt_ps, pid_ps, 
    elem_ids, xpoints_d, xface_ids, maxLoops, debug);
  assert(isFound);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("updateSurfaceDetection");
  updateSurfaceDetectionSeparate(ptcls, mesh, data_d, xpoints_d, xface_ids, iter, true);

  //updateSurfaceDetection(gp, data_d, iter, debug);
  Kokkos::Profiling::popRegion();
  //update positions and set the new element-to-particle lists
  Kokkos::Profiling::pushRegion("rebuild");
  rebuild(picparts, ptcls, elem_ids, debug);
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
  if(argc < 4)
  {
    if(comm_rank == 0)
      std::cout << "Usage: " << argv[0] 
        << " <mesh> <owners_file> <ptcls_file> "
        "[<nPtcls> <nIter> <histInterval> <timeStep>]\n";
    exit(1);
  }
  bool piscesRun = true; // add as argument later
  bool chargedTracking = false; //false for neutral tracking
  bool debug = false;

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
    std::ifstream in_str(argv[2]);
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

  std::string ptclSource = argv[3];
  if(!comm_rank)
    printf(" Particle Source file %s\n", ptclSource.c_str());
  if(!comm_rank && !chargedTracking)
    printf("WARNING: neutral particle tracking is ON \n");

  int numPtcls = 100;
  double dTime = 5e-9; //pisces:5e-9 iter 100,000
  int NUM_ITERATIONS = 10; //higher beads needs >10K
  int histInterval = 0;
  if(argc > 4)
    numPtcls = atoi(argv[4]);
  if(argc > 5)
    NUM_ITERATIONS = atoi(argv[5]);
  if(argc > 6) {
    histInterval = atoi(argv[6]);
    if(histInterval > NUM_ITERATIONS)
      histInterval = NUM_ITERATIONS;
  }
  if(argc > 7)
    dTime = atof(argv[7]);

  //TODO delete after testing
  //std::ofstream ofsHistory;
  //if(histInterval > 0)
  //  ofsHistory.open("history.txt");
  
  GitrmParticles gp(*mesh, dTime);
  // TODO use picparts 
  GitrmMesh gm(*mesh);
  if(piscesRun)
    gm.markDetectorSurfaces(true);
  //current extruded mesh has Y, Z switched
  // ramp: 330, 90, 1.5, 200,10; tgt 324, 90...; upper: 110, 0
  if(!comm_rank)
    printf("Initializing %d Particles\n", numPtcls);
  gp.initPtclsFromFile(picparts, ptclSource, numPtcls, 100, false);
  auto* ptcls = gp.ptcls;

  o::Write<o::Real>xpoints_d(3*numPtcls, 0, "xpoints");
  o::Write<o::LO>xface_ids(numPtcls, -1, "xface_ids");

  o::LO numGrid = 14;
  o::Write<o::LO>data_d(numGrid, 0);

  int nTHistory = 1;
  int dofStepData = 1;
  if(histInterval >0) {
    printf("nT_history %d %d %d interval %d \n", nTHistory, NUM_ITERATIONS, NUM_ITERATIONS/histInterval, histInterval);
    nTHistory += (int)NUM_ITERATIONS/histInterval;
    if(NUM_ITERATIONS%histInterval > 0)
      ++nTHistory;
    dofStepData = 6;
  }

  printf("\ndTime %g NUM_ITERATIONS %d nT_history %d\n", 
      dTime, NUM_ITERATIONS, nTHistory);
  assert(numPtcls*dofStepData*nTHistory > 0);
  o::Write<o::Real> ptclsDataAll(numPtcls*dofStepData);
  o::Write<o::LO> lastFilledTimeSteps(numPtcls, 0);
  o::Write<o::Real> ptclHistoryData(numPtcls*dofStepData*nTHistory);
  int iHistStep = 0;
 
  if(histInterval >0)
    updatePtclStepData(ptcls, ptclHistoryData,lastFilledTimeSteps, numPtcls, 
        dofStepData, iHistStep);
 
  fprintf(stderr, "\n*********Main Loop**********\n");
  auto end_init = std::chrono::system_clock::now();
  int iter;
  int np;
  int ps_np;
  for(iter=0; iter<NUM_ITERATIONS; iter++) {
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if(comm_rank == 0 && (debug || iter%1000 ==0))
      fprintf(stderr, "=================iter %d===============\n", iter);
   //TODO not ready for MPI
    //#if HISTORY > 0
    //o::Write<o::Real> data(numPtcls*dofStepData, -1);
    //if(iter==0 && histInterval >0)
    //  printStepData(ofsHistory, ptcls, 0, numPtcls, ptclsDataAll, data, dofStepData, true);
    //#endif

    Kokkos::Profiling::pushRegion("neutralBorisMove");
    //neutralBorisMove(ptcls, dTime);
    neutralBorisMove_float(ptcls, dTime);
    Kokkos::Profiling::popRegion();

    MPI_Barrier(MPI_COMM_WORLD);
    
    search(picparts, ptcls, gp, iter, data_d, xpoints_d, xface_ids, debug);
    if(histInterval >0) {
      //updatePtclStepData(ptcls, ptclsDataAll, numPtcls, iter+1, dofStepData, data); 
      if(iter % histInterval == 0)
        ++iHistStep;  
      updatePtclStepData(ptcls, ptclHistoryData,lastFilledTimeSteps, numPtcls, 
          dofStepData, iHistStep);
      //if((iter+1)%histInterval == 0)
        //printStepData(ofsHistory, ptcls, iter+1, numPtcls, ptclsDataAll, data, 
       // dofStepData, true); //last accum
    }
    if(comm_rank == 0 && iter%1000 ==0)
      fprintf(stderr, "nPtcls %d\n", ptcls->nPtcls());
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  
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
    gm.writeResultAsMeshTag(data_d);
  }
  if(histInterval >0)
    writePtclStepHistoryFile(ptclHistoryData, lastFilledTimeSteps, numPtcls, 
      dofStepData, nTHistory, "history.nc");
  
  Omega_h::vtk::write_parallel("mesh_vtk", mesh, picparts.dim());  
  fprintf(stderr, "done\n");
  return 0;
}


