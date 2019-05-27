#include <string>
#include "pumipic_adjacency.hpp"
#include "GitrmMesh.hpp"
#include "GitrmPush.hpp"
#include "GitrmParticles.hpp"

#include <Kokkos_Core.hpp>
#include <chrono>

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
void updatePtclPositions(SellCSigma<Particle>* scs) {
  scs->transferToDevice();
  p::kkFp3View x_scs_d("x_scs_d", scs->offsets[scs->num_slices]);
  p::hostToDeviceFp(x_scs_d, scs->getSCS<0>() );
  p::kkFp3View xtgt_scs_d("xtgt_scs_d", scs->offsets[scs->num_slices]);
  p::hostToDeviceFp(xtgt_scs_d, scs->getSCS<1>() );
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
  p::deviceToHostFp(xtgt_scs_d, scs->getSCS<1>() );
  p::deviceToHostFp(x_scs_d, scs->getSCS<0>() );
}

void rebuild(SellCSigma<Particle>* scs, o::LOs elem_ids) {
  fprintf(stderr, "rebuild\n");
  updatePtclPositions(scs);
  const int scs_capacity = scs->offsets[scs->num_slices];
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0)
      printf("elem_ids[%d] %d\n", pid, elem_ids[pid]);
  };
  scs->parallel_for(printElmIds);

  o::Write<o::LO> scs_elem_ids(scs_capacity);

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void)e;
    scs_elem_ids[pid] = elem_ids[pid];
  };
  scs->parallel_for(lamb);

  o::HostRead<o::LO> scs_elem_ids_hr(scs_elem_ids);
  int* new_element = new int[scs_capacity];
  for(int i=0; i<scs_capacity; i++) {
    new_element[i] = scs_elem_ids_hr[i];
  }
  scs->rebuildSCS(new_element);
  delete [] new_element;
}


// Copied
void search(o::Mesh& mesh, SellCSigma<Particle>* scs) {
  fprintf(stderr, "search\n");
  assert(scs->num_elems == mesh.nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->offsets[scs->num_slices];
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  bool isFound = p::search_mesh<Particle>(mesh, scs, elem_ids, maxLoops);
  assert(isFound);
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
  auto mesh = Omega_h::gmsh::read(argv[1], lib.self()); //lib.world()
  const auto r2v = mesh.ask_elem_verts();
  const auto coords = mesh.coords();

  Omega_h::Int ne = mesh.nelems();
  fprintf(stderr, "Number of elements %d \n", ne);


  GitrmMesh gm(mesh);
  Kokkos::Timer timer;

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


  GitrmParticles gp(mesh); // (const char* param_file);
  printf("\nCalculate Distance To Bdry..\n");
  gitrm_findDistanceToBdry(gp.scs, mesh, gm.bdryFaces, gm.bdryFaceInds, 
      SIZE_PER_FACE, FSKIP);

  // Put inside search
  printf("\nCalculate EField ..\n");
  gitrm_calculateE(gp.scs, mesh);
  std::cout << "\nBoris Move  \n";
  gitrm_borisMove(gp.scs, mesh, gm, 1e-6);

  fprintf(stderr, "time (seconds) %f\n", timer.seconds());
  timer.reset();

  //p::test_find_closest_point_on_triangle();

  // Omega_h::vtk::write_parallel("pisces", &mesh, mesh.dim());


  fprintf(stderr, "done\n");
  return 0;
}


