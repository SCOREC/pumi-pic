#include "pumipic_adjacency.hpp"
#include "unit_tests.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <thread>

namespace o = Omega_h;
namespace p = pumipic;

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

  if(argc != 3)
  {
    std::cout << "Usage: " << argv[0] << " <mesh> <particles per element>\n";
    exit(1);
  }

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);

  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto down_f2e = mesh.ask_down(2,1);
  const auto up_e2f = mesh.ask_up(1, 2);
  const auto up_f2r = mesh.ask_up(2, 3);
  //coordinates
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);//LOs
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  const auto dim = mesh.dim();

  /* Particle data */

  const int ppe = atoi(argv[2]);
  Omega_h::Int ne = mesh.nelems();
  const int np = ppe*ne;

  //Distribute particles to elements evenly (strat = 0)
  const int strat = 0;
  fprintf(stderr, "distribution %d-%s #elements %d #particles %d\n",
      strat, distribute_name(strat), ne, np);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  if (!distribute_particles(ne,np,strat,ptcls_per_elem, ids)) {
    return 1;
  }

  //To demonstrate push and adjacency search we store:
  //-two fp_t[3] arrays, 'Vector3d', for the current and
  // computed (pre adjacency search) positions, and
  //-an integer to store the 'new' parent element for use by
  // the particle movement procedure
  typedef MemberTypes<Vector3d, Vector3d, int > Particle;
  //'sigma', 'V', and the 'policy' control the layout of the SCS structure 
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  const bool debug = false;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 4);
  fprintf(stderr, "Sell-C-sigma C %d V %d sigma %d\n", policy.team_size(), V, sigma);
  //Create the particle structure
  SellCSigma<Particle>* scs = new SellCSigma<Particle>(policy, sigma, V, ne, np,
						       ptcls_per_elem,
						       ids, debug);
  //Set initial positions and 0 out future position
  Vector3d *initial_position_scs = scs->getSCS<0>();
  Vector3d *pushed_position_scs = scs->getSCS<1>();
  int *flag_scs = scs->getSCS<2>();

  //cleanup
  delete [] ptcls_per_elem;
  delete [] ids;
  delete scs;
  return 0;
}
