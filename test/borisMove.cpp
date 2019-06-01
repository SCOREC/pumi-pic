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

void render(o::Mesh& mesh, int iter) {
  fprintf(stderr, "%s\n", __func__);
  std::stringstream ss;
  ss << "rendered_t" << iter;
  std::string s = ss.str();
  Omega_h::vtk::write_parallel(s, &mesh, mesh.dim());
}

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
  auto mesh = Omega_h::binary::read(argv[1], world);
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
  gp.initImpurityPtcls(numPtcls, 110, 0, 1.5, 200,10);
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
  //    break;
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
    //  break;
    }
    //tagParentElements(mesh,scs,iter);
    //render(mesh,iter);
  }

  //p::test_find_closest_point_on_triangle();

  Omega_h::vtk::write_parallel("torus", &mesh, mesh.dim());


  fprintf(stderr, "done\n");
  return 0;
}


