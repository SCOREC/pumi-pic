#include "GitrmMesh.hpp"
#include "GitrmPush.hpp"
#include "GitrmParticles.hpp"
//#include "GitrmSurfaceModel.hpp"

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

void tagParentElements(o::Mesh& mesh, SCS* scs, int loop) {
  fprintf(stderr, "%s\n", __func__);
  //read from the tag
  o::LOs ehp_nm1 = mesh.get_array<o::LO>(mesh.dim(), "has_particles");
  o::Write<o::LO> ehp_nm0(ehp_nm1.size());
  auto set_ehp = OMEGA_H_LAMBDA(o::LO i) {
    ehp_nm0[i] = ehp_nm1[i];
  };
  o::parallel_for(ehp_nm1.size(), set_ehp, "set_ehp");

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void) pid;
    if(mask > 0)
      ehp_nm0[e] = loop;
  };
  scs->parallel_for(lamb);

  o::LOs ehp_nm0_r(ehp_nm0);
  mesh.set_tag(o::REGION, "has_particles", ehp_nm0_r);
}

void computeAvgPtclDensity(o::Mesh& mesh, SCS* scs){
  //create an array to store the number of particles in each element
  o::Write<o::LO> elmPtclCnt_w(mesh.nelems(),0);
  //parallel loop over elements and particles
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {

    Kokkos::atomic_fetch_add(&(elmPtclCnt_w[e]), 1);
  };
  scs->parallel_for(lamb);
  o::Write<o::Real> epc_w(mesh.nelems(),0);
  const auto convert = OMEGA_H_LAMBDA(o::LO i) {
     epc_w[i] = static_cast<o::Real>(elmPtclCnt_w[i]);
   };
  o::parallel_for(mesh.nelems(), convert, "convert_to_real");
  o::Reals epc(epc_w);
  mesh.add_tag(o::REGION, "element_particle_count", 1, o::Reals(epc));
  //get the list of elements adjacent to each vertex
  auto verts2elems = mesh.ask_up(o::VERT, mesh.dim());
  //create a device writeable array to store the computed density
  o::Write<o::Real> ad_w(mesh.nverts(),0);
  const auto accumulate = OMEGA_H_LAMBDA(o::LO i) {
    const auto deg = verts2elems.a2ab[i+1]-verts2elems.a2ab[i];
    const auto firstElm = verts2elems.a2ab[i];
    o::Real vertVal = 0.00;
    for (int j = 0; j < deg; j++){
      const auto elm = verts2elems.ab2b[firstElm+j];
      vertVal += epc[elm];
    }
    ad_w[i] = vertVal / deg;
  };
  o::parallel_for(mesh.nverts(), accumulate, "calculate_avg_density");
  o::Read<o::Real> ad_r(ad_w);
  mesh.set_tag(o::VERT, "avg_density", ad_r);
}


void rebuild(SCS* scs, o::LOs elem_ids) {
  fprintf(stderr, "rebuilding..\n");
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

void search(o::Mesh& mesh, SCS* scs, int iter, o::Write<o::LO> &data_d) {
  fprintf(stderr, "searching..\n");
  assert(scs->nElems() == mesh.nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  bool isFound = p::search_mesh<Particle>(mesh, scs, elem_ids, maxLoops);
  assert(isFound);
  //Apply surface model using face_ids, and update elem if particle reflected. 
  //elem_ids to be used in rebuild
  fprintf(stderr, "Applying surface Model..\n");
  //applySurfaceModel(mesh, scs, elem_ids);

  //output particle positions, for converting to vtk
  //storeAndPrintData(mesh, scs, iter, data_d);

  //rebuild the SCS to set the new element-to-particle lists
  rebuild(scs, elem_ids);
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
  if(argc < 2)
  {
    std::cout << "Usage: " << argv[0] 
      << " <mesh> [<BField_file>][<e_file>][prof_file][prof_density_file]\n";
    exit(1);
  }
  if(argc < 3)
  {
    fprintf(stderr, "\n ****** WARNING: No BField file provided ! \n");
  }

  std::string bFile="", eFile="", profFile="", profFileDensity="";

  if(argc >2) {
    bFile = argv[2];
  }
  if(argc >3) {
    eFile = argv[3];
  }
  if(argc >4) {
    profFile = argv[4];
  }
  if(argc > 5) {
    profFileDensity  = argv[5];
  }
  
  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::binary::read(argv[1], world);

  const auto r2v = mesh.ask_elem_verts();
  const auto coords = mesh.coords();

  Omega_h::Int ne = mesh.nelems();
  fprintf(stderr, "Number of elements %d \n", ne);

  GitrmMesh gm(mesh);

  OMEGA_H_CHECK(!profFile.empty());
  fprintf(stderr, "Adding Tags And Loadin Data\n");
  gm.addTagAndLoadData(profFile, profFileDensity);
  fprintf(stderr, "Initializing Fields and Boundary data\n");
  OMEGA_H_CHECK(!(bFile.empty() || eFile.empty()));
  gm.initEandBFields(bFile, eFile);

  fprintf(stderr, "Initializing Boundary faces\n");
  gm.initBoundaryFaces();
  fprintf(stderr, "Preprocessing Distance to boundary \n");
  // Add bdry faces to elements within 1mm
  gm.preProcessDistToBdry();
  //gm.printBdryFaceIds(false, 20);
  //gm.printBdryFacesCSR(false, 20);
  int numPtcls = 10;
  double dTime = 1e-6;//1e-7; // gitr:1e-8s for 10,000 iterations
  int NUM_ITERATIONS = 100; //000;
  fprintf(stderr, "\nInitializing %d impurity particles\n", numPtcls);
  GitrmParticles gp(mesh); // (const char* param_file);
  //current extruded mesh has Y, Z switched
  // ramp: 330, 90, 1.5, 200,10; tgt 324, 90...; upper: 110, 0
  gp.initImpurityPtclsInADir(dTime, numPtcls, 110, 0, 1.5, 200,10);

  auto &scs = gp.scs;

  o::Real dr = 0.02; //2cm 
  o::LO radGrid = (int)(2.45 - 0.8)/(2.0*dr); // x:0.8..2.45 m
  o::Write<o::LO>data_d(radGrid, 0);//*thetaGrid*phiGrid, 0);
  
  fprintf(stderr, "\ndTime %g NUM_ITERATIONS %d\n", dTime, NUM_ITERATIONS);

  mesh.add_tag(o::VERT, "avg_density", 1, o::Reals(mesh.nverts(), 0));

  fprintf(stderr, "\n*********Main Loop**********\n");
  Kokkos::Timer timer;
  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    if(scs->nPtcls() == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      fprintf(stderr, "Total iterations = %d\n", iter);
      break;
    }
    fprintf(stderr, "=================iter %d===============\n", iter);
    fprintf(stderr, "Calculating EField.. \n");
    timer.reset();
    gitrm_findDistanceToBdry(scs, mesh, gm.bdryFaces, gm.bdryFaceInds, 
      SIZE_PER_FACE, FSKIP);
    gitrm_calculateE(scs, mesh);
    fprintf(stderr, "Boris Move .. \n");
    gitrm_borisMove(scs, mesh, gm, dTime);
    //computeAvgPtclDensity(mesh, scs);
    timer.reset();
    //fprintf(stderr, "dist2bdry, E, Boris Move (seconds) %f\n", timer.seconds());
    //writeDispVectors(scs);
    search(mesh, scs, iter, data_d);
    fprintf(stderr, "time(s) %f\n", timer.seconds());
    if(scs->nPtcls() == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      fprintf(stderr, "Total iterations = %d\n", iter+1);
      break;
    }
    //tagParentElements(mesh,scs,iter);
    //render(mesh,iter);
  }
  //printGridData(data_d);
  Omega_h::vtk::write_parallel("extruded", &mesh, mesh.dim());

  fprintf(stderr, "done\n");
  return 0;
}


