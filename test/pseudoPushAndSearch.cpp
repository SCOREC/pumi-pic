#include "pumipic_adjacency.hpp"
#include "unit_tests.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <thread>

#define NUM_ITERATIONS 1

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

//To demonstrate push and adjacency search we store:
//-two fp_t[3] arrays, 'Vector3d', for the current and
// computed (pre adjacency search) positions, and
//-an integer to store the 'new' parent element for use by
// the particle movement procedure
typedef MemberTypes<Vector3d, Vector3d, int > Particle;

void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

typedef Kokkos::DefaultExecutionSpace exe_space;
typedef Kokkos::View<fp_t*, exe_space::device_type> kkFpView;
/** \brief helper function to transfer a host array to a device view */
void hostToDeviceFp(kkFpView d, fp_t* h) {
  kkFpView::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size(); ++i)
    hv(i) = h[i];
  Kokkos::deep_copy(d,hv);
}

typedef Kokkos::View<Vector3d*, exe_space::device_type> kkFp3View;
/** \brief helper function to transfer a host array to a device view */
void hostToDeviceFp(kkFp3View d, fp_t (*h)[3]) {
  kkFp3View::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size()/3; ++i) {
    hv(i,0) = h[i][0];
    hv(i,1) = h[i][1];
    hv(i,2) = h[i][2];
  }
  Kokkos::deep_copy(d,hv);
}

void push(SellCSigma<Particle>* scs, int np, fp_t distance,
    fp_t dx, fp_t dy, fp_t dz) {
  Kokkos::Timer timer;
  Vector3d *scs_initial_position = scs->getSCS<0>();
  Vector3d *scs_pushed_position = scs->getSCS<1>();
  //Move SCS data to the device
  scs->transferToDevice();

  kkFp3View position_d("position_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(position_d, scs_initial_position);
  kkFp3View new_position_d("new_position_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(new_position_d, scs_pushed_position);
  
  fp_t disp[4] = {distance,dx,dy,dz};
  kkFpView disp_d("direction_d", 4);
  hostToDeviceFp(disp_d, disp);
  fprintf(stderr, "kokkos scs host to device transfer (seconds) %f\n", timer.seconds());

#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)  
  double totTime = 0;
  timer.reset();
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void) e;
    fp_t dir[3];
    dir[0] = disp_d(0)*disp_d(1);
    dir[1] = disp_d(0)*disp_d(2);
    dir[2] = disp_d(0)*disp_d(3);
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      new_position_d(pid,0) = position_d(pid,0) * dir[0];
      new_position_d(pid,1) = position_d(pid,1) * dir[1];
      new_position_d(pid,2) = position_d(pid,2) * dir[2];
    });
  });
  totTime += timer.seconds();
  printTiming("scs push", totTime);
#endif
}

void setInitialPtclCoords(Vector3d* p, int numPtcls) {
  const fp_t insetFaceDiameter = 0.6;
  const fp_t insetFacePlane = 0.20001; // just above the inset bottom face
  const fp_t insetFaceRim = 0.3; // in x and z
  const fp_t insetFaceCenter = 0; // in x and z
  const fp_t x_delta = insetFaceDiameter / numPtcls;
  for(int i=0; i<numPtcls; i++) {
    p[i][0] = insetFaceRim + (x_delta * i);
    p[i][1] = insetFacePlane;
    p[i][2] = insetFaceCenter;
  }
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
  const auto r2v = mesh.ask_elem_verts();
  const auto coords = mesh.coords();

  /* Particle data */
  const int numPtcls = 12;

  //Distribute particles to elements evenly (strat = 0)
  Omega_h::Int ne = mesh.nelems();
  const int strat = 0;
  fprintf(stderr, "distribution %d-%s #elements %d #particles %d\n",
      strat, distribute_name(strat), ne, numPtcls);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  if (!distribute_particles(ne,numPtcls,strat,ptcls_per_elem, ids)) {
    return 1;
  }

  //'sigma', 'V', and the 'policy' control the layout of the SCS structure 
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  const bool debug = false;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 4);
  fprintf(stderr, "Sell-C-sigma C %d V %d sigma %d\n", policy.team_size(), V, sigma);
  //Create the particle structure
  SellCSigma<Particle>* scs = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
						       ptcls_per_elem,
						       ids, debug);
  //Set initial positions and 0 out future position
  Vector3d *initial_position_scs = scs->getSCS<0>();
  setInitialPtclCoords(initial_position_scs, numPtcls); //TODO

  //run search to move the particles to their starting elements //TODO
 
  //rebuild the SCS to set the new element-to-particle lists //TODO

  int *flag_scs = scs->getSCS<2>();
  (void)flag_scs; //TODO

  //define parameters controlling particle motion
  //move the particles in +y direction by 1/20th of the
  //pisces model's height
  fp_t distance = .05;
  fp_t dx = 0;
  fp_t dy = 1;
  fp_t dz = 0;

  Kokkos::Timer timer;
  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    fprintf(stderr, "\n");
    timer.reset();
    push(scs, numPtcls, distance, dx, dy, dz);
    fprintf(stderr, "kokkos scs with macros push and transfer (seconds) %f\n", timer.seconds());
  }

  //cleanup
  delete [] ptcls_per_elem;
  delete [] ids;
  delete scs;
  return 0;
}
