#include "GitrmMesh.hpp"
#include "GitrmPush.hpp"
#include "GitrmParticles.hpp"
#include "GitrmIonizeRecombine.hpp"
//#include "GitrmSurfaceModel.hpp"
#include "Omega_h_file.hpp"

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
  scs->parallel_for(updatePtclPos);
}


void rebuild(SCS* scs, o::LOs elem_ids) {
  //fprintf(stderr, "rebuilding..\n");
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto pid_d =  scs->get<2>();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0 ) {//&& elem_ids[pid] >= 0) 
      //printf(">> Particle remains %d \n", pid);
      printf("rebuild:elem_ids[%d] %d ptcl %d\n", pid, elem_ids[pid], pid_d(pid));
    }
  };
 // scs->parallel_for(printElmIds);

  SCS::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void)e;
    scs_elem_ids(pid) = elem_ids[pid];
  };
  scs->parallel_for(lamb);
  
  scs->rebuild(scs_elem_ids);
}

void search(GitrmParticles& gp, GitrmMesh& gm, int iter, o::Write<o::LO>& data_d,
  GitrmIonizeRecombine& gir, bool debug=false) {
  auto& mesh = gm.mesh;
  SCS* scs = gp.scs;
  assert(scs->nElems() == mesh.nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  o::Write<o::Real>xpoints_d(3*scsCapacity, 0, "xpoints");
  o::Write<o::LO>xface_ids(scsCapacity, -1, "xface_ids");
  auto x_scs = scs->get<0>();
  auto xtgt_scs = scs->get<1>();
  auto pid_scs = scs->get<2>();

  bool isFound = p::search_mesh<Particle>(mesh, scs, x_scs, xtgt_scs, pid_scs, 
    elem_ids, xpoints_d, xface_ids, maxLoops);
  assert(isFound);

  gp.collisionPoints = o::Reals(xpoints_d);
  gp.collisionPointFaceIds = o::LOs(xface_ids);
  auto elm_ids = o::LOs(elem_ids);
  // skipped for neutral particle tracking 
  
  if(gir.chargedPtclTracking) {
    gitrm_ionize(scs, gir, gp, gm, elm_ids, debug);
    gitrm_recombine(scs, gir, gp, gm, elm_ids, debug);
  }

  //Apply surface model using face_ids, and update elem if particle reflected. 
  //elem_ids to be used in rebuild
  //fprintf(stderr, "Applying surface Model..\n");
  //applySurfaceModel(mesh, scs, elem_ids);

  //output particle positions, for converting to vtk
  storePiscesData(mesh, gp, iter, data_d, true);
  //update positions and set the new element-to-particle lists
  rebuild(scs, elem_ids);
}

void profileAndInterpolateTest(GitrmMesh& gm, bool debug=false) {
  gm.printDensityTempProfile(0.05, 20, 0.7, 2);
  //gm.test_interpolateFields(true);
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
  // TODO use paramter file
  if(argc < 7)
  {
    std::cout << "Usage: " << argv[0] 
      << " <mesh><Bfile><prof_file><ptcls_file><rate_file>"
      << "[<nPtcls><nIter><timeStep>]\n";
    exit(1);
  }

  std::string bFile="", profFile="", ptclSource="", ionizeRecombFile="";
  bool piscesRun = true;
  bool debug = false;
  bool chargedTracking = false; //false for neutral tracking
  
  if(!chargedTracking)
    printf("WARNING: neutral particle tracking is ON \n");

  bool storePtclsGrids = false;
  o::Real shiftB = 0; //pisces=0; otherwise 1.6955; 
  if(piscesRun)
    shiftB = 0; //1.6955;

  bFile = argv[2];
  profFile = argv[3];
  ptclSource  = argv[4];
  ionizeRecombFile = argv[5];
  printf(" Mesh file %s\n", argv[1]);
  printf(" Particle Source file %s\n", ptclSource.c_str());
  printf(" Profile file %s\n", profFile.c_str());
  printf(" IonizeRecomb File %s\n", ionizeRecombFile.c_str());
  auto mesh = Omega_h::read_mesh_file(argv[1], lib.self());
  printf("Number of elements %d verts %d\n", mesh.nelems(), mesh.nverts());

  GitrmMesh gm(mesh);

  if(piscesRun)
    gm.markDetectorCylinder(true);

  printf("Initializing Fields and Boundary data\n");
  OMEGA_H_CHECK(!bFile.empty());
  gm.initBField(bFile, shiftB);

  std::cout << "done E,B \n";

  printf("Adding Tags And Loadin Data %s\n", profFile.c_str());
  OMEGA_H_CHECK(!profFile.empty());
  gm.addTagAndLoadData(profFile, profFile);

  OMEGA_H_CHECK(!ionizeRecombFile.empty());
  GitrmIonizeRecombine gir(ionizeRecombFile, chargedTracking);

  printf("Initializing Boundary faces\n");
  gm.initBoundaryFaces();
  printf("Preprocessing Distance to boundary \n");
  // Add bdry faces to elements within 1mm
  gm.preProcessDistToBdry();
  int numPtcls = 0;
  double dTime = 5e-9; //pisces:5e-9 for 100,000 iterations
  int NUM_ITERATIONS = 10000; //higher beads needs >10K
  if(argc > 6)
    numPtcls = atoi(argv[6]);
  if(argc > 7)
    NUM_ITERATIONS = atoi(argv[7]);
  if(argc > 8)
    dTime = atof(argv[8]);

  GitrmParticles gp(mesh, dTime);
  //current extruded mesh has Y, Z switched
  // ramp: 330, 90, 1.5, 200,10; tgt 324, 90...; upper: 110, 0
  printf("Initializing Particles\n");

  if(ptclSource.empty())
    gp.initImpurityPtclsInADir(numPtcls, 110, 0, 1.5, 200,10);
  else
    gp.initImpurityPtclsFromFile(ptclSource, numPtcls, 100, false);

  auto &scs = gp.scs;

  if(debug)
    profileAndInterpolateTest(gm, true); //move to unit_test

  o::LO numGrid = 14;
  o::Write<o::LO>data_d(numGrid, 0);

  o::LO ptclGrids = 20;
  o::Write<o::GO>ptclDataR(ptclGrids, 0);
  o::Write<o::GO>ptclDataZ(ptclGrids, 0);
  printf("\ndTime %g NUM_ITERATIONS %d\n", dTime, NUM_ITERATIONS);

  mesh.add_tag(o::VERT, "avg_density", 1, o::Reals(mesh.nverts(), 0));
  Omega_h::vtk::write_parallel("meshvtk", &mesh, mesh.dim());

  fprintf(stderr, "\n*********Main Loop**********\n");

  auto start_main = std::chrono::system_clock::now(); 
  Kokkos::Timer timer;
  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    if(scs->nPtcls() == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      fprintf(stderr, "Total iterations = %d\n", iter);
      break;
    }
    fprintf(stderr, "=================iter %d===============\n", iter);

    if(gir.chargedPtclTracking) {    
      gitrm_findDistanceToBdry(gp, gm);
      gitrm_calculateE(gp, mesh, debug);
      gitrm_borisMove(scs, gm, dTime);
    }
    else
      neutralBorisMove(scs,dTime);

    search(gp, gm, iter, data_d, gir, debug);

    if(storePtclsGrids) {
      storePtclDataInGridsRZ(scs, iter, ptclDataR, ptclGrids, 1);
      storePtclDataInGridsRZ(scs, iter, ptclDataZ, 1, ptclGrids);
    }
    if(iter%100 ==0)
      fprintf(stderr, "nPtcls %d\n", scs->nPtcls());
    if(scs->nPtcls() == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      fprintf(stderr, "Total iterations = %d\n", iter+1);
      break;
    }
  }
  auto end_sim = std::chrono::system_clock::now();
  std::chrono::duration<double> dur_sec = end_sim - start_main;
  std::cout << "Main duration " << dur_sec.count()/60 << " min.\n";
  std::chrono::duration<double> dur_sec2 = start_main - start_sim;
  std::cout << "Pre-procesing duration " << dur_sec2.count()/60 << " min.\n";
  std::cout << "Profiles in R direction \n";
  if(storePtclsGrids) {
    printGridData(ptclDataR);
    std::cout << "Profiles in Z direction \n";
    printGridData(ptclDataZ);
    std::cout << "Pisces detections \n";
    printGridData(data_d);
  }
  fprintf(stderr, "done\n");

  return 0;
}


