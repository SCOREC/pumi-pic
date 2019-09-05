#include <Omega_h_mesh.hpp>
#include <SCS_Macros.h>
#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pseudoXGCmTypes.hpp"
#include "gyroScatter.hpp"
#include <fstream>
#include "ellipticalPush.hpp"
#include <random>

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  OMEGA_H_CHECK(argc == 2);
  const auto strict = true;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  assert(deviceCount==1);
  if (!rank) {
    printf("device count per process %d\n", deviceCount);
    printf("world ranks %d\n", size);
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

  if(!rank)
    fprintf(stderr, "%d ranks loading mesh %s\n", size, argv[1]);
  auto mesh = Omega_h::binary::read(argv[1], lib.self(), strict);
  if(!rank)
    fprintf(stderr, "mesh tri %d\n", mesh.nelems());
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  const auto vtx_to_elm = mesh.ask_up(0,2);
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  const auto edge_to_elm = mesh.ask_up(1,2);
  OMEGA_H_CHECK(cudaSuccess == cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank)
    fprintf(stderr, "done\n");
  return 0;
}
