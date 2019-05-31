#include <string>
#include "GitrmMesh.hpp"
#include "GitrmPush.hpp"
#include "GitrmParticles.hpp"

#include "Omega_h_mesh.hpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_adjacency.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include "pumipic_library.hpp"

using particle_structs::fp_t;
using particle_structs::lid_t;
using particle_structs::Vector3d;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;
using particle_structs::distribute_name;
using particle_structs::elemCoords;

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

//TODO this update has to be in Boris Move ?, or kept track of inside it
void updatePtclPositions(SCS* scs) {
  auto x_scs_d = scs->get<0>();
  auto xtgt_scs_d = scs->get<1>();
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      x_scs_d(pid,0) = xtgt_scs_d(pid,0);
      x_scs_d(pid,1) = xtgt_scs_d(pid,1);
      x_scs_d(pid,2) = xtgt_scs_d(pid,2);
      xtgt_scs_d(pid,0) = 0;
      xtgt_scs_d(pid,1) = 0;
      xtgt_scs_d(pid,2) = 0;
    });
  });
}


void rebuild(SCS* scs, o::LOs elem_ids) {
  fprintf(stderr, "rebuild\n");
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0)
      printf("elem_ids[%d] %d\n", pid, elem_ids[pid]);
  };
  scs->parallel_for(printElmIds);

  SCS::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void)e;
    scs_elem_ids(pid) = elem_ids[pid];
  };
  scs->parallel_for(lamb);
  
  scs->rebuild(scs_elem_ids);
}

void search(o::Mesh& mesh, SCS* scs) {
  fprintf(stderr, "search\n");
  assert(scs->num_elems == mesh.nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  bool isFound = p::search_mesh<Particle>(mesh, scs, elem_ids, maxLoops);
  assert(isFound);
  //rebuild the SCS to set the new element-to-particle lists
  rebuild(scs, elem_ids);
}

void push(SCS* scs, int np, fp_t distance,
    fp_t dx, fp_t dy, fp_t dz) {
  fprintf(stderr, "push\n");


  Kokkos::Timer timer;
  auto position_d = scs->get<0>();
  auto new_position_d = scs->get<1>();

  const auto capacity = scs->capacity();

  fp_t disp[4] = {distance,dx,dy,dz};
  p::kkFpView disp_d("direction_d", 4);
  p::hostToDeviceFp(disp_d, disp);
  fprintf(stderr, "kokkos scs host to device transfer (seconds) %f\n", timer.seconds());

  o::Write<o::Real> ptclUnique_d(capacity, 0);

  double totTime = 0;
  timer.reset();
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    fp_t dir[3];
    dir[0] = disp_d(0)*disp_d(1);
    dir[1] = disp_d(0)*disp_d(2);
    dir[2] = disp_d(0)*disp_d(3);
    new_position_d(pid,0) = position_d(pid,0) + dir[0] + ptclUnique_d[pid];
    new_position_d(pid,1) = position_d(pid,1) + dir[1] + ptclUnique_d[pid];
    new_position_d(pid,2) = position_d(pid,2) + dir[2] + ptclUnique_d[pid];
  };
  scs->parallel_for(lamb);

  totTime += timer.seconds();
  printTiming("scs push", totTime);
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
      << " <mesh> [<BField_file>][<e_file>][prof_file][prof_file_density]\n";
    exit(1);
  }
  if(argc < 3)
  {
    std::cout << "\n\n ****** WARNING: No BField file provided ! \n\n\n";
  }

  std::string bFile, eFile, profFile, profFileDensity;
  bFile = eFile = profFile = profFileDensity = "";

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
  auto mesh = Omega_h::gmsh::read(argv[1], world);
  const auto r2v = mesh.ask_elem_verts();
  const auto coords = mesh.coords();

  Omega_h::Int ne = mesh.nelems();
  fprintf(stderr, "Number of elements %d \n", ne);


  GitrmMesh gm(mesh);

  OMEGA_H_CHECK(!profFile.empty());
  std::cout << "\n adding Tags And Loadin Data ..\n";
  gm.addTagAndLoadData(profFile, profFileDensity);

  std::cout << "\nInitialize Fields and Boundary data..\n";
  OMEGA_H_CHECK(!(bFile.empty() || eFile.empty()));
  gm.initEandBFields(bFile, eFile);

  std::cout << "\nInitialize Boundary faces ...\n";
  gm.initBoundaryFaces();

  std::cout << "\nPreprocessing Distance to boundary ...\n";
  // Add bdry faces to elements within 1mm
  gm.preProcessDistToBdry();
  //gm.printBdryFaceIds(false, 20);
  //gm.printBdryFacesCSR(false, 20);

  int numPtcls = 10;
  double dTime = 1e-8;
  int NUM_ITERATIONS = 10;


  GitrmParticles gp(mesh, 10); // (const char* param_file);
  gp.initImpurityPtcls(numPtcls, 110, 0, 1.5, 5);
  auto &scs = gp.scs;
  printf("\nCalculate Distance To Bdry..\n");
  gitrm_findDistanceToBdry(gp.scs, mesh, gm.bdryFaces, gm.bdryFaceInds, 
      SIZE_PER_FACE, FSKIP);

  // Put inside search
  printf("\nCalculate EField ..\n");
  gitrm_calculateE(gp.scs, mesh);

  Kokkos::Timer timer;
  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    if(scs->num_ptcls == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    fprintf(stderr, "iter %d\n", iter);
    //computeAvgPtclDensity(mesh, scs);
    timer.reset();
    fprintf(stderr, "Boris Move: dTime=%.5f\n", dTime);
    gitrm_borisMove(gp.scs, mesh, gm, dTime);
    fprintf(stderr, "push and transfer (seconds) %f\n", timer.seconds());
    //writeDispVectors(scs);
    timer.reset();
    search(mesh, gp.scs);
    fprintf(stderr, "search, rebuild, and transfer (seconds) %f\n", timer.seconds());
    if(scs->num_ptcls == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    //tagParentElements(mesh,scs,iter);
    //render(mesh,iter);
  }

  //p::test_find_closest_point_on_triangle();

  Omega_h::vtk::write_parallel("torus", &mesh, mesh.dim());


  fprintf(stderr, "done\n");
  return 0;
}


