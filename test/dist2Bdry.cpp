#include "pumipic_adjacency.hpp"
#include "GitrmMesh.hpp"
#include "GitrmMesh.hpp"
#include "unit_tests.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <thread>

#define NUM_ITERATIONS 20

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
typedef Kokkos::View<lid_t*, exe_space::device_type> kkLidView;
void hostToDeviceLid(kkLidView d, lid_t *h) {
  kkLidView::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size(); ++i) {
    hv(i) = h[i];
  }
  Kokkos::deep_copy(d,hv);
}

void deviceToHostLid(kkLidView d, lid_t *h) {
  kkLidView::HostMirror hv = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(hv,d);
  for(size_t i=0; i<hv.size(); ++i) {
    h[i] = hv(i);
  }
}

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

void setRand(SellCSigma<Particle>* scs, kkFpView disp_d, o::Write<o::Real> rand_d) {
  srand (time(NULL));
  kkLidView pid_d("pid_d", scs->offsets[scs->num_slices]);
  hostToDeviceLid(pid_d, scs->getSCS<2>() );
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void) e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      rand_d[pid] = (disp_d(0))*((double)(std::rand())/RAND_MAX - 1.0);
      fprintf(stderr, "rand ptcl %d %.3f\n", pid_d(pid), rand_d[pid]);
    });
  });
}

void push(SellCSigma<Particle>* scs, int np, fp_t distance,
    fp_t dx, fp_t dy, fp_t dz, bool rand=false) {
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

  o::Write<o::Real> ptclUnique_d(scs->offsets[scs->num_slices], 0);
  if(rand) {
    fprintf(stderr, "random ptcl movement enabled\n");
    setRand(scs, disp_d, ptclUnique_d);
  }

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
      new_position_d(pid,0) = position_d(pid,0) + dir[0] + ptclUnique_d[pid];
      new_position_d(pid,1) = position_d(pid,1) + dir[1] + std::abs(ptclUnique_d[pid]);
      new_position_d(pid,2) = position_d(pid,2) + dir[2] + ptclUnique_d[pid];
    });
  });
  totTime += timer.seconds();
  printTiming("scs push", totTime);
#endif
  deviceToHostFp(new_position_d, scs_pushed_position);
}


void updatePtclPositions(SellCSigma<Particle>* scs) {
  scs->transferToDevice();
  kkFp3View x_scs_d("x_scs_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(x_scs_d, scs->getSCS<0>() );
  kkFp3View xtgt_scs_d("xtgt_scs_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(xtgt_scs_d, scs->getSCS<1>() );
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
  deviceToHostFp(xtgt_scs_d, scs->getSCS<1>() );
  deviceToHostFp(x_scs_d, scs->getSCS<0>() );
}

void rebuild(SellCSigma<Particle>* scs, o::LOs elem_ids) {
  fprintf(stderr, "rebuild\n");
  updatePtclPositions(scs); 

  const int scs_capacity = scs->offsets[scs->num_slices];
  o::Write<o::LO> scs_elem_ids(scs_capacity);
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      scs_elem_ids[pid] = elem_ids[pid];
    });
  });
  o::HostRead<o::LO> scs_elem_ids_hr(scs_elem_ids);
  int* new_element = new int[scs_capacity];
  for(int i=0; i<scs_capacity; i++) {
    new_element[i] = scs_elem_ids_hr[i];
  }
  scs->rebuildSCS(new_element);
  delete [] new_element;
}

void search(o::Mesh& mesh, SellCSigma<Particle>* scs) {
  fprintf(stderr, "search\n");

  assert(scs->num_elems == mesh.nelems());

  //define the 20+ input args...
  //TODO create the mesh arrays inside the function
  //TODO document the search_mesh function args after cleanup
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

  //boundary classification and coordinates
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);//LOs

  //flags
  Omega_h::Write<Omega_h::LO> ptcl_flags(scs->num_ptcls, 1);         // < 0 - particle has hit a boundary or reached its destination
  Omega_h::Write<Omega_h::LO> elem_ids(scs->num_ptcls,-1);           // TODO use scs
  Omega_h::Write<Omega_h::LO> coll_adj_face_ids(scs->num_ptcls, -1); // why is this needed outside the search fn? what is it?
  Omega_h::Write<Omega_h::Real> bccs(4*scs->num_ptcls, -1.0);        // TODO use scs. for debugging only?
  Omega_h::Write<Omega_h::Real> xpoints(3*scs->num_ptcls, -1.0);     // what is this? for debugging only?

  Omega_h::Write<Omega_h::LO> pids(scs->num_ptcls,-1);

  //set particle positions and parent element ids
  scs->transferToDevice();
  kkFp3View x_scs_d("x_scs_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(x_scs_d, scs->getSCS<0>() );
  kkFp3View xtgt_scs_d("xtgt_scs_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(xtgt_scs_d, scs->getSCS<1>() );

  kkLidView pid_d("pid_d", scs->offsets[scs->num_slices]);
  hostToDeviceLid(pid_d, scs->getSCS<2>() );

  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      x0[pid] = x_scs_d(pid,0);
      y0[pid] = x_scs_d(pid,1);
      z0[pid] = x_scs_d(pid,2);
      x[pid] = xtgt_scs_d(pid,0);
      y[pid] = xtgt_scs_d(pid,1);
      z[pid] = xtgt_scs_d(pid,2);
      elem_ids[pid] = e;
      pids[pid] = pid_d(pid);
    });
  });

  // sanity check
  auto f = OMEGA_H_LAMBDA(o::LO i) {
    printf("elem_ids[%d] %d %d %f %f %f -> %f %f %f\n", i, pids[i], elem_ids[i], x0[i], y0[i], z0[i], x[i], y[i], z[i]);
  };
  o::parallel_for(scs->num_ptcls, f, "print_x");

  Omega_h::LO loops = 0;
  Omega_h::LO maxLoops = 100;
 /*
  bool isFound = p::search_mesh(
      pids, nelems, x0, y0, z0, x, y, z,
      dual, down_r2f,
      side_is_exposed, mesh2verts, coords, face_verts,
      ptcl_flags, elem_ids, coll_adj_face_ids, bccs,
      xpoints, loops, maxLoops);
  assert(isFound);
*/
  //rebuild the SCS to set the new element-to-particle lists
  rebuild(scs, elem_ids);
}

//HACK to avoid having an unguarded comma in the SCS PARALLEL macro
OMEGA_H_INLINE o::Matrix<3, 4> gatherVectors(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

void setPtclIds(SellCSigma<Particle>* scs) {
  fprintf(stderr, "%s\n", __func__);
  scs->transferToDevice();
  kkLidView pid_d("pid_d", scs->offsets[scs->num_slices]);
  hostToDeviceLid(pid_d, scs->getSCS<2>() );
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      pid_d(pid) = pid;
    });
  });
  deviceToHostLid(pid_d, scs->getSCS<2>() );
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
  fprintf(stderr, "%s\n", __func__);
  const fp_t insetFaceDiameter = 0.5;
  const fp_t insetFacePlane = 0.20001; // just above the inset bottom face
  const fp_t insetFaceRim = -0.25; // in x
  const fp_t insetFaceCenter = 0; // in x and z
  fp_t x_delta = insetFaceDiameter / (numPtcls-1);
  if( numPtcls == 1 )
    x_delta = 0;
  for(int i=0; i<numPtcls; i++) {
    p[i][0] = insetFaceCenter;
    p[i][1] = insetFacePlane;
    p[i][2] = insetFaceRim + (x_delta * i);
  }
}

void render(o::Mesh& mesh, int iter) {
  printf("in render\n");
  std::stringstream ss;
  ss << "rendered_t" << iter;
  std::string s = ss.str();
  Omega_h::vtk::write_parallel(s, &mesh, mesh.dim());
}

void computeAvgPtclDensity(o::Mesh& mesh, SellCSigma<Particle>* scs){
  //transfer the SCS structure to the device
  scs->transferToDevice();
  //create an array to store the number of particles in each element
  o::Write<o::LO> elmPtclCnt_w(mesh.nelems(),0);
  //parallel loop over elements and particles
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    //get the omega_h element id
    int o_e = row_to_element(e);
    (void)o_e;
    int ptcls = 0;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      (void)pid; //silence warning
      ptcls++;
    });
    elmPtclCnt_w[o_e] = ptcls;
  });
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

  Omega_h::Int ne = mesh.nelems();
  fprintf(stderr, "number of elements %d \n", ne);



  //associate one floating point value with each mesh vertex and set them to 0
//  mesh.add_tag(o::VERT, "avg_density", 1, o::Reals(mesh.nverts(), 0));

  GitrmMesh gm(mesh);
  Kokkos::Timer timer;
  // Add bdry faces to elements within 1mm

  //preProcessDistToBdry(mesh, bdryFacesW, numBdryFaceIds, bdryFaceIds, bdryFlags);
  gm.preProcessDistToBdry();
  gm.printBdryFaceIds(false, 20); //numBdryFaceIds, mesh);  
  gm.convert2ReadOnlyCSR();
  gm.printBdryFacesCSR(false, 23);


 /* Particle data */
  const int numPtcls = 12;  //how many particles in domain
  const int initialElement = 271;

  fprintf(stderr, "number of elements %d number of particles %d\n",
      ne, numPtcls);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  for(int i=0; i<ne; i++)
    ptcls_per_elem[i] = 0;
  for(int i=0; i<numPtcls; i++)
    ids[initialElement].push_back(i);
  ptcls_per_elem[initialElement] = numPtcls;

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
  setPtclIds(scs);

  p::findDistanceToBdry(scs, r2v, coords, gm.bdryFaces, gm.bdryFaceInds, 
      SIZE_PER_FACE, FSKIP, ne);

  fprintf(stderr, "time (seconds) %f\n", timer.seconds());
  timer.reset();

  delete [] ids;
  delete scs;

  fprintf(stderr, "done\n");
  return 0;
}


