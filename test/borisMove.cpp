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

void rebuild(SCS* scs, o::LOs elem_ids) {
  //fprintf(stderr, "rebuilding..\n");
  //updatePtclPositions(scs);
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

void search(GitrmParticles& gp, GitrmMesh& gm, int iter, o::Write<o::LO> &data_d,
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
  gitrm_ionize(scs, gir, gp, gm, elm_ids, debug);
  gitrm_recombine(scs, gir, gp, gm, elm_ids, debug);

  //Apply surface model using face_ids, and update elem if particle reflected. 
  //elem_ids to be used in rebuild
  //fprintf(stderr, "Applying surface Model..\n");
  //applySurfaceModel(mesh, scs, elem_ids);

  //output particle positions, for converting to vtk
  storeAndPrintData(mesh, gp, iter, data_d, true);
  //rebuild the SCS to set the new element-to-particle lists
  rebuild(scs, elem_ids);
}
//TODO move test_interpolateTet() from ionizeRecombine
void unit_test(SCS* scs, GitrmMesh& gm, bool debug=false) {
  printf("\nInterpolation Test only for init ptcl's element vertices\n");
  auto& mesh = gm.mesh;
  const auto densIon_d = gm.densIon_d;
  auto x0 = gm.densIonX0;
  auto z0 = gm.densIonZ0;
  auto nx = gm.densIonNx;
  auto nz = gm.densIonNz;
  auto dx = gm.densIonDx;
  auto dz = gm.densIonDz;
  const auto densVtx = mesh.get_array<o::Real>(o::VERT, "IonDensityVtx");
  test_interpolateTet(scs, mesh, densVtx, densIon_d, x0, z0, dx, dz, nx, nz, true);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc,argv);
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
      << " <mesh><Bfile><Efiel><prof_file><ptcls_file><rate_file>"
      << "[<nPtcls><nIter>]\n";
    exit(1);
  }

  std::string bFile="", eFile="", profFile="", ptclSource="", ionizeRecombFile="";
  bool piscesRun = false;

  //TODO
  piscesRun = true;
  o::Real shiftE = 0;
  o::Real shiftB = 0;
  if(piscesRun) {
    shiftE = 1.6955;
    shiftB = 0; //TODO
  }

  bFile = argv[2];
  eFile = argv[3];
  profFile = argv[4];
  ptclSource  = argv[5];
  ionizeRecombFile = argv[6];

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::read_mesh_file(argv[1], world);
  printf("Number of elements %d verts %d\n", mesh.nelems(), mesh.nverts());

  GitrmMesh gm(mesh);

  if(piscesRun)
    gm.markDetectorCylinder(true);

  printf("Initializing Fields and Boundary data\n");
  OMEGA_H_CHECK(!(bFile.empty()||eFile.empty()));
  gm.initEandBFields(bFile, eFile, shiftB, shiftE);

  std::cout << "done E,B \n";

  printf("Adding Tags And Loadin Data %s\n", profFile.c_str());
  OMEGA_H_CHECK(!profFile.empty());
  gm.addTagAndLoadData(profFile, profFile);

  OMEGA_H_CHECK(!ionizeRecombFile.empty());
  GitrmIonizeRecombine gir(ionizeRecombFile);

  printf("Initializing Boundary faces\n");
  gm.initBoundaryFaces();
  printf("Preprocessing Distance to boundary \n");
  // Add bdry faces to elements within 1mm
  gm.preProcessDistToBdry();
  int numPtcls = 0;
  double dTime = 1e-8; //gitr:1e-8s for 10,000 iterations
  int NUM_ITERATIONS = 10000; //higher beads needs >10K
  if(argc > 7)
    numPtcls = atoi(argv[7]);
  if(argc > 8)
    NUM_ITERATIONS = atoi(argv[8]);

  GitrmParticles gp(mesh, dTime);
  //current extruded mesh has Y, Z switched
  // ramp: 330, 90, 1.5, 200,10; tgt 324, 90...; upper: 110, 0
  printf("Initializing Particles\n");

  if(ptclSource.empty())
    gp.initImpurityPtclsInADir(numPtcls, 110, 0, 1.5, 200,10);
  else
    gp.initImpurityPtclsFromFile(ptclSource, numPtcls, 100, false);

  auto &scs = gp.scs;

  //unit_test(scs, gm, true); //move to unit_test

  // o::LO radGrid = (int)(2.45 - 0.8)/(2.0*dr); // x:0.8..2.45 m
  o::LO numGrid = 14;
  o::Write<o::LO>data_d(numGrid, 0);//*thetaGrid*phiGrid, 0);
  
  printf("\ndTime %g NUM_ITERATIONS %d\n", dTime, NUM_ITERATIONS);

  mesh.add_tag(o::VERT, "avg_density", 1, o::Reals(mesh.nverts(), 0));
  Omega_h::vtk::write_parallel("meshvtk", &mesh, mesh.dim());

  fprintf(stderr, "\n*********Main Loop**********\n");
  auto start_sim = std::chrono::system_clock::now(); 
  Kokkos::Timer timer;
  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    if(scs->nPtcls() == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      fprintf(stderr, "Total iterations = %d\n", iter);
      break;
    }
    fprintf(stderr, "=================iter %d===============\n", iter);
    gitrm_findDistanceToBdry(gp, gm);
    //mesh, gm.bdryFaces, gm.bdryFaceInds, SIZE_PER_FACE, FSKIP);
    gitrm_calculateE(gp, mesh);
    gitrm_borisMove(scs, gm, dTime);
    //computeAvgPtclDensity(mesh, scs);
    //writeDispVectors(scs);
    timer.reset();
    search(gp, gm, iter, data_d, gir );

    if(iter%100 ==0)
      fprintf(stderr, "time(s) %f nPtcls %d\n", timer.seconds(), scs->nPtcls());
    if(scs->nPtcls() == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      fprintf(stderr, "Total iterations = %d\n", iter+1);
      break;
    }
    //tagParentElements(mesh,scs,iter);
    //render(mesh,iter);
  }
  auto end_sim = std::chrono::system_clock::now();
  std::chrono::duration<double> dur_sec = end_sim - start_sim;
  std::cout << "Simulation duration " << dur_sec.count()/60 << " min.\n";
  printGridData(data_d);
  fprintf(stderr, "done\n");

  return 0;
}


