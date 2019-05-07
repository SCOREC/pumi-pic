#include "Omega_h_mesh.hpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_adjacency.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>

#define NUM_ITERATIONS 30

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
//-an integer to store the particles id
typedef MemberTypes<Vector3d, Vector3d, int> Particle;


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

void writeDispVectors(SellCSigma<Particle>* scs) {
  fprintf(stderr, "%s\n", __func__);
  scs->transferToDevice();

  const int capacity = scs->offsets[scs->num_slices];
  p::kkFp3View x_nm1("x_nm1", capacity);
  p::hostToDeviceFp(x_nm1, scs->getSCS<0>());
  p::kkFp3View x_nm0("x_nm0", capacity);
  p::hostToDeviceFp(x_nm0, scs->getSCS<1>());
  p::kkLidView pid_d("pid", capacity);
  p::hostToDeviceLid(pid_d, scs->getSCS<2>());
  o::Write<o::Real> px_nm1(capacity*3);
  o::Write<o::Real> px_nm0(capacity*3);
  o::Write<o::LO> pid_w(capacity);

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    pid_w[pid] = -1;
    if(mask > 0) {
      pid_w[pid] = pid_d(pid);
      for(int i=0; i<3; i++) {
        px_nm1[pid*3+i] = x_nm1(pid,i);
        px_nm0[pid*3+i] = x_nm0(pid,i);
      }
    }
  };
  scs->parallel_for(lamb);

  o::HostRead<o::Real> px_nm0_hr(px_nm0);
  o::HostRead<o::Real> px_nm1_hr(px_nm1);
  o::HostRead<o::LO> pid_hr(pid_w);
  for(int i=0; i< capacity; i++) {
    if(pid_hr[i] != -1) {
      fprintf(stderr, "ptclID%d  %.3f %.3f %.3f initial\n",
        pid_hr[i], px_nm1_hr[i*3+0], px_nm1_hr[i*3+1], px_nm1_hr[i*3+2]);
    }
  }
  for(int i=0; i< capacity; i++) {
    if(pid_hr[i] != -1) {
      fprintf(stderr, "ptclID%d  %.3f %.3f %.3f final\n",
        pid_hr[i], px_nm0_hr[i*3+0], px_nm0_hr[i*3+1], px_nm0_hr[i*3+2]);
    }
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

  const auto capacity = scs->offsets[scs->num_slices];
  p::kkFp3View position_d("position_d", capacity);
  p::hostToDeviceFp(position_d, scs_initial_position);
  p::kkFp3View new_position_d("new_position_d", capacity);
  p::hostToDeviceFp(new_position_d, scs_pushed_position);
  
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

  p::deviceToHostFp(new_position_d, scs_pushed_position);
}

void tagParentElements(o::Mesh& mesh, SellCSigma<Particle>* scs, int loop) {
  fprintf(stderr, "%s\n", __func__);
  //read from the tag
  o::LOs ehp_nm1 = mesh.get_array<o::LO>(mesh.dim(), "has_particles");
  o::Write<o::LO> ehp_nm0(ehp_nm1.size());
  auto set_ehp = OMEGA_H_LAMBDA(o::LO i) {
    ehp_nm0[i] = ehp_nm1[i];
  };
  o::parallel_for(ehp_nm1.size(), set_ehp, "set_ehp");
  scs->transferToDevice();

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void) pid;
    if(mask > 0)
      ehp_nm0[e] = loop;
  };
  scs->parallel_for(lamb);

  o::LOs ehp_nm0_r(ehp_nm0);
  mesh.set_tag(o::REGION, "has_particles", ehp_nm0_r);
}

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

//HACK to avoid having an unguarded comma in the SCS PARALLEL macro
OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

void setPtclIds(SellCSigma<Particle>* scs) {
  fprintf(stderr, "%s\n", __func__);
  scs->transferToDevice();
  p::kkLidView pid_d("pid_d", scs->offsets[scs->num_slices]);
  p::hostToDeviceLid(pid_d, scs->getSCS<2>() );
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      pid_d(pid) = pid;
    });
  });
  p::deviceToHostLid(pid_d, scs->getSCS<2>() );
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
  p::kkFp3View x_scs_d("x_scs_d", scs->offsets[scs->num_slices]);
  p::hostToDeviceFp(x_scs_d, scs->getSCS<0>() );
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
    auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
    auto center = average(cell_nodes2coords);
    if(mask > 0) {
      printf("elm %d xyz %f %f %f\n", e, center[0], center[1], center[2]);
      for(int i=0; i<3; i++)
        x_scs_d(pid,i) = center[i];
    }
  };
  scs->parallel_for(lamb);
  p::deviceToHostFp(x_scs_d, scs->getSCS<0>() );
}

void setTargetPtclCoords(SellCSigma<Particle>* scs) {
  fprintf(stderr, "%s\n", __func__);
  scs->transferToDevice();
  const auto capacity = scs->offsets[scs->num_slices];
  p::kkFp3View xtgt_scs_d("xtgt_scs_d", capacity);
  p::hostToDeviceFp(xtgt_scs_d, scs->getSCS<1>());
  const fp_t insetFaceDiameter = 0.5;
  const fp_t insetFacePlane = 0.201; // just above the inset bottom face
  const fp_t insetFaceRim = -0.25; // in x
  const fp_t insetFaceCenter = 0; // in x and z
  fp_t x_delta = insetFaceDiameter / (capacity-1);
  printf("x_delta %.4f\n", x_delta);
  if( scs->num_ptcls == 1 )
    x_delta = 0;
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      xtgt_scs_d(pid,0) = insetFaceCenter;
      xtgt_scs_d(pid,1) = insetFacePlane;
      xtgt_scs_d(pid,2) = insetFaceRim + (x_delta * pid);
    }
  };
  scs->parallel_for(lamb);
  p::deviceToHostFp(xtgt_scs_d, scs->getSCS<1>());
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
  o::LOs foo(ne, 1, "foo");
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
  //Create the particle structure
  SellCSigma<Particle>* scs = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
						       ptcls_per_elem,
						       ids, debug);
  delete [] ptcls_per_elem;

  //Set initial and target positions so search will
  // find the parent elements
  setInitialPtclCoords(mesh, scs);
  setPtclIds(scs);
  setTargetPtclCoords(scs);

  //run search to move the particles to their starting elements
  search(mesh,scs);
 
  //define parameters controlling particle motion
  //move the particles in +y direction by 1/20th of the
  //pisces model's height
  fp_t heightOfDomain = 1.0;
  fp_t distance = heightOfDomain/20;
  fp_t dx = -0.5;
  fp_t dy = 0.8;
  fp_t dz = 0;

  fprintf(stderr, "push distance %.3f push direction %.3f %.3f %.3f\n",
      distance, dx, dy, dz);

  o::LOs elmTags(ne, 0, "elmTagVals");
  mesh.add_tag(o::REGION, "has_particles", 1, elmTags);

  Kokkos::Timer timer;
  for(int iter=0; iter<NUM_ITERATIONS; iter++) {
    if(scs->num_ptcls == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    fprintf(stderr, "iter %d\n", iter);
    timer.reset();
    push(scs, scs->num_ptcls, distance, dx, dy, dz);
    fprintf(stderr, "push and transfer (seconds) %f\n", timer.seconds());
    writeDispVectors(scs);
    timer.reset();
    search(mesh,scs);
    fprintf(stderr, "search, rebuild, and transfer (seconds) %f\n", timer.seconds());
    if(scs->num_ptcls == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    tagParentElements(mesh,scs,iter);
    render(mesh,iter);
  }

  //cleanup
  delete [] ids;
  delete scs;

  Omega_h::vtk::write_parallel("rendered", &mesh, mesh.dim());
  fprintf(stderr, "done\n");
  return 0;
}
