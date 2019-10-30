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
#include "GitrmIonizeRecombine.hpp"
//#include "GitrmSurfaceModel.hpp"


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

void rebuild(p::Mesh& picparts, SCS* scs, o::LOs elem_ids, 
    const bool output=false) {
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
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
}

void search(p::Mesh& picparts, GitrmParticles& gp, GitrmMesh& gm,
    GitrmIonizeRecombine& gir, int iter, o::Write<o::LO>& data_d, 
    bool debug=false) {
  //auto& picparts = gm.picparts;
  o::Mesh* mesh = picparts.mesh();
  Kokkos::Profiling::pushRegion("gitrm_search");
  SCS* scs = gp.scs;
  assert(scs->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 20;
  const auto scsCapacity = scs->capacity();
  assert(scsCapacity > 0);
  o::Write<o::LO> elem_ids(scsCapacity,-1);

  auto x_scs = scs->get<0>();
  auto xtgt_scs = scs->get<1>();
  auto pid_scs = scs->get<2>();

  bool isFound = p::search_mesh_3d<Particle>(*mesh, scs, x_scs, xtgt_scs, pid_scs, 
    elem_ids, gp.collisionPoints, gp.collisionPointFaceIds, maxLoops, debug);
  assert(isFound);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("updateGitrmData");
  
  if(gir.chargedPtclTracking) {
    gitrm_ionize(scs, gir, gp, gm, elem_ids, debug);
    gitrm_recombine(scs, gir, gp, gm, elem_ids, debug);
  }

  //Apply surface model using face_ids, and update elem if particle reflected. 
  //fprintf(stderr, "Applying surface Model..\n");
  //applySurfaceModel(mesh, scs, elem_ids);
  bool resetFids = true;
  storePiscesData(mesh, gp, data_d, iter, resetFids, true);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("rebuild");
  //update positions and set the new element-to-particle lists
  rebuild(picparts, scs, elem_ids, debug);
  Kokkos::Profiling::popRegion();
}

void profileAndInterpolateTest(GitrmMesh& gm, bool debug=false, bool inter=false) {
  gm.printDensityTempProfile(0.05, 20, 0.7, 2);
  if(inter)
    gm.test_interpolateFields(true);
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
  if(argc < 6)
  {
    if(comm_rank == 0)
      std::cout << "Usage: " << argv[0] 
        << " <mesh> <owners_file> <ptcls_file> <prof_file> <rate_file>"
        << " [<nPtcls><nIter> <histInterval> <gitrDataFileName> ]\n";
    exit(1);
  }
  bool piscesRun = true; // add as argument later
  bool chargedTracking = true; //false for neutral tracking
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
  std::string profFile = argv[4];
  std::string ionizeRecombFile = argv[5];
  std::string bFile=""; //TODO

  if (!comm_rank) {
    if(!chargedTracking)
      printf("WARNING: neutral particle tracking is ON \n");
    printf(" Mesh file %s\n", argv[1]);
    printf(" Particle Source file %s\n", ptclSource.c_str());
    printf(" Profile file %s\n", profFile.c_str());
    printf(" IonizeRecomb File %s\n", ionizeRecombFile.c_str());
  }
  int numPtcls = 100;
  int histInterval = 0;
  double dTime = 5e-9; //pisces:5e-9 for 100,000 iterations
  int NUM_ITERATIONS = 10; //higher beads needs >10K
  
  if(argc > 6)
    numPtcls = atoi(argv[6]);
  if(argc > 7)
    NUM_ITERATIONS = atoi(argv[7]);
  if(argc > 8) {
    histInterval = atoi(argv[8]);
    if(histInterval >NUM_ITERATIONS)
      histInterval = NUM_ITERATIONS;
  }
  std::string gitrDataFileName = "";
  if(argc > 9)
    gitrDataFileName = argv[9];

  GitrmParticles gp(*mesh, dTime);
  // TODO use picparts 
  GitrmMesh gm(*mesh);
  if(piscesRun)
    gm.markPiscesCylinder(true);
  //current extruded mesh has Y, Z switched
  // ramp: 330, 90, 1.5, 200,10; tgt 324, 90...; upper: 110, 0
  if(!comm_rank)
    printf("Initializing Particles\n");
  gp.initPtclsFromFile(picparts, ptclSource, numPtcls, 100, false);

  int compare_with_gitr = 0;
  int testNumPtcls = 10;
  int testNumPtclsRead = 0;
  if(compare_with_gitr) {
    assert(COMPARE_WITH_GITR == 1);
    gp.readGITRPtclStepDataNcFile(gitrDataFileName, testNumPtcls, testNumPtclsRead);
  }
  auto* scs = gp.scs;
  const auto scsCapacity = scs->capacity();

  if(!piscesRun) {
    std::string bFile = "get_from_argv";
    double shiftB = 0; // non-pisces= 1.6955
    gm.initBField(bFile, shiftB); 
  }

  printf("Adding Tags And Loading Profile Data %s\n", profFile.c_str());
  OMEGA_H_CHECK(!profFile.empty());
  gm.addTagAndLoadProfileData(profFile, profFile);

  OMEGA_H_CHECK(!ionizeRecombFile.empty());
  GitrmIonizeRecombine gir(ionizeRecombFile, chargedTracking);

  printf("Initializing Boundary faces\n");
  gm.initBoundaryFaces(false);
  printf("Preprocessing Distance to boundary \n");
  // Add bdry faces to elements within 1mm
  gm.preProcessDistToBdry();

  if(debug)
    profileAndInterpolateTest(gm, true); //move to unit_test

  o::LO numGrid = 14;
  o::Write<o::LO>data_d(numGrid, 0);

  printf("\ndTime %g NUM_ITERATIONS %d\n", dTime, NUM_ITERATIONS);

  int nTHistory = 1;
  int dofStepData = 1; //TODO see if 0 OK for no-history
  if(histInterval >0) {
    nTHistory += (int)NUM_ITERATIONS/histInterval;
    if(NUM_ITERATIONS % histInterval)
      ++nTHistory;
    printf("nHistory %d histInterval %d\n", nTHistory, histInterval);
    dofStepData = 6;
  }
  //always assert size>0 for device data init
  assert(numPtcls*dofStepData*nTHistory >0);
  o::Write<o::Real> ptclsDataAll(numPtcls*dofStepData); // TODO delete
  o::Write<o::Real> ptclHitoryData(numPtcls*dofStepData*nTHistory);
  int iHistStep = 0;
  if(histInterval >0)
    updatePtclStepData(scs, ptclHitoryData, numPtcls, dofStepData, iHistStep);
  
  fprintf(stderr, "\n*********Main Loop**********\n");
  auto end_init = std::chrono::system_clock::now();
  int np;
  int scs_np;
  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    scs_np = scs->nPtcls();
    MPI_Allreduce(&scs_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if(comm_rank == 0 && (debug || iter%1000 ==0))
      fprintf(stderr, "=================iter %d===============\n", iter);
    Kokkos::Profiling::pushRegion("BorisMove");
    if(gir.chargedPtclTracking) {
      gitrm_findDistanceToBdry(gp, gm);
      gitrm_calculateE(gp, *mesh, debug);
      gitrm_borisMove(scs, gm, dTime);
    }
    else
      neutralBorisMove(scs,dTime);
    Kokkos::Profiling::popRegion();
    MPI_Barrier(MPI_COMM_WORLD);

    search(picparts, gp, gm, gir, iter, data_d, debug);
    
    if(histInterval >0) {
      // move-over if. 0th step(above) kept; last step available at the end.
      if(iter % histInterval == 0)
        ++iHistStep;        
      updatePtclStepData(scs, ptclHitoryData, numPtcls, dofStepData, iHistStep);
    }
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
    gm.markPiscesCylinderResult(data_d);
  }
  if(histInterval >0) {
    printf("writing out history file \n");
    writePtclStepHistoryNcFile(ptclHitoryData, numPtcls, dofStepData, 
      nTHistory, "history.nc");
  }

  Omega_h::vtk::write_parallel("meshvtk", mesh, mesh->dim());

  fprintf(stderr, "done\n");
  return 0;
}




