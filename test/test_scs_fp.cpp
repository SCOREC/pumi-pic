#include "testMesh.hpp"
#include "testParticles.hpp"

#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_mesh.hpp"
#include <Kokkos_Core.hpp>

namespace o = Omega_h;
namespace p = pumipic;

using particle_structs::fp_t;
using particle_structs::lid_t;

void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
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


  if(argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " <mesh>\n";
    exit(1);
  }

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  //doesn't work for extruded mesh
  auto mesh = Omega_h::read_mesh_file(argv[1], world);

  //doesn't work for gmsh
  //auto mesh = Omega_h::binary::read(argv[1], world);
  Omega_h::Int ne = mesh.nelems();
  fprintf(stderr, "Number of elements %d \n", ne);

  testMesh gm(mesh);

  int numPtcls = 10;
  fprintf(stderr, "\nInitializing %d impurity particles\n", numPtcls);
  testParticles gp(mesh); // (const char* param_file);

  gp.testInitImpurityPtclsInADir(numPtcls);

  fprintf(stderr, "done\n");
  return 0;
}
