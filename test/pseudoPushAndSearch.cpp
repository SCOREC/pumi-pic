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

void deviceToHostFp(kkFp3View d, fp_t (*h)[3]) {
  kkFp3View::HostMirror hv = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(hv,d);
  for(size_t i=0; i<hv.size()/3; ++i) {
    h[i][0] = hv(i,0);
    h[i][1] = hv(i,1);
    h[i][2] = hv(i,2);
  }
}

void push(SellCSigma<Particle>* scs, int np, fp_t distance,
    fp_t dx, fp_t dy, fp_t dz) {
  fprintf(stderr, "push\n");
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

void search(o::Mesh& mesh, SellCSigma<Particle>* scs) {
  fprintf(stderr, "search\n");

  assert(scs->num_elems == mesh.nelems());

  //define the 20+ input args...
  //TODO create the mesh arrays inside the function
  //TODO document the search_mesh function args after cleanup
  Omega_h::LO something = 1; 
  Omega_h::Int nelems = mesh.nelems();

  //initial positions
  Omega_h::Write<Omega_h::Real> x0(scs->num_ptcls,0);
  Omega_h::Write<Omega_h::Real> y0(scs->num_ptcls,0);
  Omega_h::Write<Omega_h::Real> z0(scs->num_ptcls,0);
  //final positions
  Omega_h::Write<Omega_h::Real> x(scs->num_ptcls,0);
  Omega_h::Write<Omega_h::Real> y(scs->num_ptcls,0);
  Omega_h::Write<Omega_h::Real> z(scs->num_ptcls,0);


  //mesh adjacencies
  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto down_f2e = mesh.ask_down(2,1);
  const auto up_e2f = mesh.ask_up(1, 2);
  const auto up_f2r = mesh.ask_up(2, 3);

  //boundary classification and coordinates
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);//LOs

  //flags
  Omega_h::Write<Omega_h::LO> ptcl_flags(scs->num_ptcls, 1);         // TODO what does this store?
  Omega_h::Write<Omega_h::LO> elem_ids(scs->num_ptcls,-1);           // TODO use scs
  Omega_h::Write<Omega_h::LO> coll_adj_face_ids(scs->num_ptcls, -1); // why is this needed outside the search fn? what is it?
  Omega_h::Write<Omega_h::Real> bccs(4*scs->num_ptcls, -1.0);        // TODO use scs. for debugging only?
  Omega_h::Write<Omega_h::Real> xpoints(3*scs->num_ptcls, -1.0);     // what is this? for debugging only?

  //set particle positions and parent element ids
  scs->transferToDevice();
  kkFp3View x_scs_d("x_scs_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(x_scs_d, scs->getSCS<0>() );
  kkFp3View xtgt_scs_d("xtgt_scs_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(xtgt_scs_d, scs->getSCS<1>() );
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    printf("elm %d\n", e);
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      printf("ptcl %d\n", pid);
      x0[pid] = x_scs_d(pid,0);
      y0[pid] = x_scs_d(pid,1);
      z0[pid] = x_scs_d(pid,2);
      x[pid] = xtgt_scs_d(pid,0);
      y[pid] = xtgt_scs_d(pid,1);
      z[pid] = xtgt_scs_d(pid,2);
      elem_ids[pid] = e;
    });
  });

  // sanity check
  auto f = OMEGA_H_LAMBDA(o::LO i) {
    printf("elem_ids[%d] %d %f %f %f -> %f %f %f\n", i, elem_ids[i], x0[i], y0[i], z0[i], x[i], y[i], z[i]);
  };
  o::parallel_for(scs->num_ptcls, f, "print_x");

  Omega_h::LO loops = 0;
  Omega_h::LO maxLoops = 100;
  bool isFound = p::search_mesh(
      something, nelems, x0, y0, z0, x, y, z,
      dual, down_r2f, down_f2e, up_e2f, up_f2r,
      side_is_exposed, mesh2verts, coords, face_verts,
      ptcl_flags, elem_ids, coll_adj_face_ids, bccs,
      xpoints, loops, maxLoops);
  assert(isFound);
  auto printElmIds = OMEGA_H_LAMBDA(o::LO i) {
    printf("elem_ids[%d] %d\n", i, elem_ids[i]);
  };
  o::parallel_for(scs->num_ptcls, printElmIds, "print_elm_ids");
  o::Write<o::LO> elm_has_ptcls(mesh.nelems(), -1);
  auto setElmsWithPtcls = OMEGA_H_LAMBDA(o::LO i) {
    elm_has_ptcls[ elem_ids[i] ] = 1;
  };
  o::parallel_for(scs->num_ptcls, setElmsWithPtcls, "setElmsWithPtcls");
  o::LOs ehp(elm_has_ptcls);
  mesh.add_tag(o::REGION, "initial_elms", 1, ehp);
}

//HACK to avoid having an unguarded comma in the SCS PARALLEL macro
OMEGA_H_INLINE o::Matrix<3, 4> gatherVectors(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

void setInitialPtclCoords(o::Mesh& mesh, SellCSigma<Particle>* scs) {
  //get centroid of parent element and set the child particle coordinates
  //most of this is copied from Omega_h_overlay.cpp get_cell_center_location
  //It isn't clear why the template parameter for gather_[verts|vectors] was
  //sized eight... maybe something to do with the 'Overlay'.  Given that there
  //are four vertices bounding a tet, I'm setting that parameter to four below.
  auto cells2nodes = mesh.get_adj(o::REGION, o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  //set particle positions and parent element ids
  scs->transferToDevice();
  kkFp3View x_scs_d("x_scs_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(x_scs_d, scs->getSCS<0>() );
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
    auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
    auto center = average(cell_nodes2coords);
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      printf("elm %d xyz %f %f %f\n", e, center[0], center[1], center[2]);
      for(int i=0; i<3; i++)
        x_scs_d(pid,i) = center[i];
    });
  });
  deviceToHostFp(x_scs_d, scs->getSCS<0>() );
}

void setTargetPtclCoords(Vector3d* p, int numPtcls) {
  const fp_t insetFaceDiameter = 0.5;
  const fp_t insetFacePlane = 0.20001; // just above the inset bottom face
  const fp_t insetFaceRim = -0.25; // in x
  const fp_t insetFaceCenter = 0; // in x and z
  const fp_t x_delta = insetFaceDiameter / (numPtcls-1);
  for(int i=0; i<numPtcls; i++) {
    p[i][0] = insetFaceCenter;
    p[i][1] = insetFacePlane;
    p[i][2] = insetFaceRim + (x_delta * i);
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

  if(argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " <mesh>\n";
    exit(1);
  }

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);
  const auto r2v = mesh.ask_elem_verts();
  const auto coords = mesh.coords();

  /* Particle data */
  const int numPtcls = 12;
  const int initialElement = 271;

  Omega_h::Int ne = mesh.nelems();
  fprintf(stderr, "number of elements %d number of particles %d\n",
      ne, numPtcls);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  for(int i=0; i<ne; i++)
    ptcls_per_elem[i] = 0;
  for(int i=0; i<numPtcls; i++)
    ids[initialElement].push_back(i);
  ptcls_per_elem[initialElement] = numPtcls;
  for(int i=0; i<ne; i++)
    if(ptcls_per_elem[i]>0)
      printf("ppe[%d] %d\n", i, ptcls_per_elem[i]);

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
  delete [] ptcls_per_elem;

  //Set initial and target positions so search will
  // find the parent elements
  setInitialPtclCoords(mesh, scs);
  Vector3d *final_position_scs = scs->getSCS<1>();
  setTargetPtclCoords(final_position_scs, numPtcls);

  //run search to move the particles to their starting elements
  search(mesh,scs);
 
  //rebuild the SCS to set the new element-to-particle lists //TODO

  /*
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
  */

  //cleanup
  delete [] ids;
  delete scs;

  Omega_h::vtk::write_parallel("rendered", &mesh, mesh.dim());
  return 0;
}
