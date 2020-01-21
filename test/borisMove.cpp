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
#include "GitrmSurfaceModel.hpp"


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

void rebuild(p::Mesh& picparts, PS* ptcls, o::LOs elem_ids, 
    const bool output=false) {
  updatePtclPositions(ptcls);
  const int ps_capacity = ptcls->capacity();
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
}

void search(p::Mesh& picparts, GitrmParticles& gp, GitrmMesh& gm,
    GitrmIonizeRecombine& gir, GitrmSurfaceModel& sm, int iter, 
    o::Write<o::LO>& data_d, bool debug=false) {
  //auto& picparts = gm.picparts;
  o::Mesh* mesh = picparts.mesh();
  Kokkos::Profiling::pushRegion("gitrm_search");
  PS* ptcls = gp.ptcls;
  assert(ptcls->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 200;
  const auto psCapacity = ptcls->capacity();
  assert(psCapacity > 0);
  o::Write<o::LO> elem_ids(psCapacity,-1);

  auto x_ps = ptcls->get<0>();
  auto xtgt_ps = ptcls->get<1>();
  auto pid_ps = ptcls->get<2>();

  bool isFound = p::search_mesh_3d<Particle>(*mesh, ptcls, x_ps, xtgt_ps, pid_ps, 
    elem_ids, gp.collisionPoints, gp.collisionPointFaceIds, maxLoops, debug);
  assert(isFound);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("updateGitrmData");
  
  if(gir.chargedPtclTracking) {
    gitrm_ionize(ptcls, gir, gp, gm, elem_ids, false);
    gitrm_recombine(ptcls, gir, gp, gm, elem_ids, false);
    //gitrm_surfaceReflection_test(ptcls, sm, gp, gm, elem_ids, false);
  }
  bool resetFids = true;
  storePiscesData(mesh, gp, data_d, iter, resetFids, false);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("rebuild");
  //update positions and set the new element-to-particle lists
  rebuild(picparts, ptcls, elem_ids, debug);
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
  if(argc < 7)
  {
    if(comm_rank == 0)
      std::cout << "Usage: " << argv[0] 
        << " <mesh> <owners_file> <ptcls_file> <prof_file> <rate_file><surf_file>"
        << " [<nPtcls><nIter> <histInterval> <gitrDataInFileName> ]\n";
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
  std::string surfModelFile = argv[6];

  if (!comm_rank) {
    if(!chargedTracking)
      printf("WARNING: neutral particle tracking is ON \n");
    printf(" Mesh file %s\n", argv[1]);
    printf(" Particle Source file %s\n", ptclSource.c_str());
    printf(" Profile file %s\n", profFile.c_str());
    printf(" IonizeRecomb File %s\n", ionizeRecombFile.c_str());
  }
  int numPtcls = 1;
  int histInterval = 0;
  double dTime = 5e-9; //pisces:5e-9 for 100,000 iterations
  int NUM_ITERATIONS = 1; //higher beads needs >10K
  
  if(argc > 7)
    numPtcls = atoi(argv[7]);
  if(argc > 8)
    NUM_ITERATIONS = atoi(argv[8]);
  if(argc > 9) {
    histInterval = atoi(argv[9]);
    if(histInterval > NUM_ITERATIONS)
      histInterval = NUM_ITERATIONS;
  }
  std::string gitrDataFileName;
  if(argc > 10)
    gitrDataFileName = argv[10];
  if (!comm_rank)
    printf(" gitr comparison DataFile %s\n", gitrDataFileName.c_str());
  
  GitrmParticles gp(*mesh, dTime);
  // TODO use picparts 
  GitrmMesh gm(*mesh);

  if(CREATE_GITR_MESH) {
    gm.createSurfaceGitrMesh();
  }

  if(piscesRun)
    gm.markPiscesCylinder(true);
  //current extruded mesh has Y, Z switched
  // ramp: 330, 90, 1.5, 200,10; tgt 324, 90...; upper: 110, 0
  if(!comm_rank)
    printf("Initializing Particles\n");
  gp.initPtclsFromFile(picparts, ptclSource, numPtcls, 100, false);

  int useGitrRandNums = USE_GITR_RND_NUMS;
  int testNumPtcls = 1;
  int testRead = 0;
  if(useGitrRandNums) {
    gp.readGITRPtclStepDataNcFile(gitrDataFileName, testNumPtcls, testRead);
    assert(testNumPtcls >= numPtcls);
  }
  auto* ptcls = gp.ptcls;
  const auto psCapacity = ptcls->capacity();

  if(!piscesRun) {
    std::string bFile = "get_from_argv";
    double shiftB = 0; // non-pisces= 1.6955
    gm.initBField(bFile, shiftB); 
  }

  printf("Adding Tags And Loading Profile Data %s\n", profFile.c_str());
  OMEGA_H_CHECK(!profFile.empty());
  auto initFields = gm.addTagsAndLoadProfileData(profFile, profFile);

  OMEGA_H_CHECK(!ionizeRecombFile.empty());
  GitrmIonizeRecombine gir(ionizeRecombFile, chargedTracking);

  printf("Initializing Boundary faces\n");
  auto initBdry = gm.initBoundaryFaces(initFields, false);
  printf("Preprocessing: dist-to-boundary faces\n");
  int nD2BdryTetSubDiv = D2BDRY_GRIDS_PER_TET;
  int readInCsrBdryData = USE_READIN_CSR_BDRYFACES;
  if(readInCsrBdryData) {
    gm.readDist2BdryFacesData("bdryFaces_in.nc");
  } else {
    gm.preprocessSelectBdryFacesFromAll(initBdry);
  }
  bool writeTextBdryFaces = WRITE_TEXT_D2BDRY_FACES;
  if(writeTextBdryFaces)
    gm.writeBdryFacesDataText(nD2BdryTetSubDiv);  
  bool writeBdryFaceCoords = WRITE_BDRY_FACE_COORDS_NC;
  if(writeBdryFaceCoords)
    gm.writeBdryFaceCoordsNcFile(2); //selected  
  bool writeMeshFaceCoords = WRITE_MESH_FACE_COORDS_NC;
  if(writeMeshFaceCoords)
    gm.writeBdryFaceCoordsNcFile(1); //all
  int writeBdryFacesFile = WRITE_OUT_BDRY_FACES_FILE;
  if(writeBdryFacesFile && !readInCsrBdryData) {
    std::string bdryOutName = "bdryFaces_" + 
      std::to_string(nD2BdryTetSubDiv) + "div.nc"; 
    gm.writeDist2BdryFacesData(bdryOutName, nD2BdryTetSubDiv);
  }

  GitrmSurfaceModel sm(gm, surfModelFile);

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
  o::Write<o::LO> lastFilledTimeSteps(numPtcls, 0);
  o::Write<o::Real> ptclHistoryData(numPtcls*dofStepData*nTHistory);
  int iHistStep = 0;
  if(histInterval >0)
    updatePtclStepData(ptcls, ptclHistoryData, lastFilledTimeSteps, numPtcls, 
      dofStepData, iHistStep);
  
  fprintf(stderr, "\n*********Main Loop**********\n");
  auto end_init = std::chrono::system_clock::now();
  int np;
  int ps_np;

  //TODO replace by Kokkos
  std::srand(time(NULL));

  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if(comm_rank == 0)// && (debug || iter%1000 ==0))
      fprintf(stderr, "=================iter %d===============\n", iter);
    Kokkos::Profiling::pushRegion("BorisMove");
    if(gir.chargedPtclTracking) {
      gitrm_findDistanceToBdry(gp, gm, 0);
      gitrm_calculateE(gp, *mesh, false, gm);
      gitrm_borisMove(ptcls, gm, dTime, false);
    }
    else
      neutralBorisMove(ptcls,dTime);
    Kokkos::Profiling::popRegion();
    MPI_Barrier(MPI_COMM_WORLD);

    search(picparts, gp, gm, gir, sm, iter, data_d, debug);
    
    if(histInterval >0) {
      // move-over if. 0th step(above) kept; last step available at the end.
      if(iter % histInterval == 0)
        ++iHistStep;        
      updatePtclStepData(ptcls, ptclHistoryData, lastFilledTimeSteps, numPtcls,
        dofStepData, iHistStep);
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
  std::cout << "\nInitialization duration " << dur_init.count()/60 << " min.\n";
  std::chrono::duration<double> dur_steps = end_sim - end_init;
  std::cout << "Total Main Loop duration " << dur_steps.count()/60 << " min.\n";
  
  if(piscesRun) {
    std::string fname("piscesCounts.txt");
    printGridData(data_d, fname, "piscesDetected");
    gm.markPiscesCylinderResult(data_d);
  }
  if(histInterval >0) {
    writePtclStepHistoryFile(ptclHistoryData, lastFilledTimeSteps, numPtcls, 
      dofStepData, nTHistory, "gitrm-history.nc");
  }
  
  Omega_h::vtk::write_parallel("meshvtk", mesh, mesh->dim());

  fprintf(stderr, "Done\n");
  return 0;
}




