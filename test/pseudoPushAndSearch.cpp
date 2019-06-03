#include "Omega_h_mesh.hpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_adjacency.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include "pumipic_mesh.hpp"
#include <fstream>
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
typedef SellCSigma<Particle> SCS;

void render(o::Mesh& mesh, int iter, int comm_rank) {
  std::stringstream ss;
  ss << "pseudoPush_t" << iter<<"_r"<<comm_rank;
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

void writeDispVectors(SCS* scs) {
  const int capacity = scs->capacity();
  auto x_nm1 = scs->get<0>();
  auto x_nm0 = scs->get<1>();
  auto pid_d = scs->get<2>();
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

void push(SCS* scs, int np, fp_t distance,
    fp_t dx, fp_t dy, fp_t dz) {
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

void tagParentElements(o::Mesh& mesh, SCS* scs, int loop) {
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

void rebuild(SCS* scs, o::LOs elem_ids, const bool output) {
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(output && mask > 0)
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

void search(o::Mesh& mesh, SCS* scs, bool output) {
  assert(scs->num_elems == mesh.nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  bool isFound = p::search_mesh<Particle>(mesh, scs, elem_ids, maxLoops);
  assert(isFound);
  //rebuild the SCS to set the new element-to-particle lists
  rebuild(scs, elem_ids, output );
}

//HACK to avoid having an unguarded comma in the SCS PARALLEL macro
OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

void setPtclIds(SCS* scs) {
  auto pid_d = scs->get<2>();
  PS_PARALLEL_FOR_ELEMENTS(scs, thread, e, {
    (void)e;
    PS_PARALLEL_FOR_PARTICLES(scs, thread, pid, {
      pid_d(pid) = pid;
    });
  });
}

void setInitialPtclCoords(o::Mesh& mesh, SCS* scs, bool output) {
  //get centroid of parent element and set the child particle coordinates
  //most of this is copied from Omega_h_overlay.cpp get_cell_center_location
  //It isn't clear why the template parameter for gather_[verts|vectors] was
  //sized eight... maybe something to do with the 'Overlay'.  Given that there
  //are four vertices bounding a tet, I'm setting that parameter to four below.
  auto cells2nodes = mesh.get_adj(o::REGION, o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  //set particle positions and parent element ids
  auto x_scs_d = scs->get<0>();
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
    auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
    auto center = average(cell_nodes2coords);
    if(mask > 0) {
      if (output)
        printf("elm %d xyz %f %f %f\n", e, center[0], center[1], center[2]);
      for(int i=0; i<3; i++)
        x_scs_d(pid,i) = center[i];
    }
  };
  scs->parallel_for(lamb);
}

//Sunflower algorithm adapted from: https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
void setSunflowerPositions(SCS* scs, const fp_t insetFaceDiameter, const fp_t insetFacePlane,
                           const fp_t insetFaceRim, const fp_t insetFaceCenter) {
  const fp_t insetFaceRadius = insetFaceDiameter/2;
  auto xtgt_scs_d = scs->get<1>();
  const o::LO n = scs->capacity();
  const fp_t phi = (sqrt(5) + 1) / 2;
  auto setPoints = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    const fp_t r = sqrt(pid + 0.5) / sqrt(n - 1 / 2);
    const fp_t theta = 2 * M_PI * pid / (phi*phi);
    xtgt_scs_d(pid, 0) = insetFaceCenter + insetFaceRadius * r * cos(theta);
    xtgt_scs_d(pid, 1) = insetFacePlane;
    xtgt_scs_d(pid, 2) = insetFaceCenter + insetFaceRadius * r * sin(theta);
  };
  scs->parallel_for(setPoints);
}
void setLinearPositions(SCS* scs, const fp_t insetFaceDiameter, const fp_t insetFacePlane,
                        const fp_t insetFaceRim, const fp_t insetFaceCenter){
  auto xtgt_scs_d = scs->get<1>();
  fp_t x_delta = insetFaceDiameter / (scs->capacity()-1);
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
}
void setTargetPtclCoords(SCS* scs) {
  const fp_t insetFaceDiameter = 0.5;
  const fp_t insetFacePlane = 0.201; // just above the inset bottom face
  const fp_t insetFaceRim = -0.25; // in x
  const fp_t insetFaceCenter = 0; // in x and z
  setSunflowerPositions(scs, insetFaceDiameter, insetFacePlane, insetFaceRim, insetFaceCenter);
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
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int min_args = 2 + (comm_size > 1);
  if(argc < min_args || argc > min_args + 2) {
    if (!comm_rank) {
      if (comm_size == 1)
        std::cout << "Usage: " << argv[0] << " <mesh> [numPtcls (12)] [initial element (271)]\n";
      else
        std::cout << "Usage: " << argv[0] << " <mesh> <owner_file> [numPtcls (12)] "
          "[initial element (271)]\n";
    }
    exit(1);
  }
  if (comm_rank == 0) {
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
  auto full_mesh = Omega_h::gmsh::read(argv[1], lib.self());
  Omega_h::HostWrite<Omega_h::LO> host_owners(full_mesh.nelems());
  if (comm_size > 1) {
    std::ifstream in_str(argv[2]);
    if (!in_str) {
      if (!comm_rank)
        fprintf(stderr,"Cannot open file %s\n", argv[2]);
      return EXIT_FAILURE;
    }
    int own;
    int index = 0;
    while(in_str >> own) 
      host_owners[index++] = own;
  }
  else
    for (int i = 0; i < full_mesh.nelems(); ++i)
      host_owners[i] = 0;
  Omega_h::Write<Omega_h::LO> owner(host_owners);
  
  //Create Picparts with the full mesh
  pumipic::Mesh picparts(full_mesh,owner);

  Omega_h::Mesh& mesh = *(picparts.mesh());
  mesh.ask_elem_verts();
  mesh.coords();
  if (comm_rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh.nverts(), mesh.nedges(), 
           mesh.nfaces(), mesh.nelems());

  /* Particle data */
  const int numPtcls = (argc >=min_args + 1) ? atoi(argv[min_args]) : 12;
  const int initialElement = (argc >= min_args + 2) ? atoi(argv[min_args+1]) : 271;
  const bool output = numPtcls <= 12 && (comm_rank == 0);
  Omega_h::Int ne = mesh.nelems();
  if (comm_rank == 0)
    fprintf(stderr, "number of elements %d number of particles %d\n",
            ne, numPtcls);
  o::LOs foo(ne, 1, "foo");
  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  //Element gids is left empty since there is no partitioning of the mesh yet
  SCS::kkGidView element_gids;
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    ptcls_per_elem(i) = 0;
    if (i == initialElement)
      ptcls_per_elem(i) = numPtcls;
  });
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
    if (output && np > 0)
      printf("ppe[%d] %d\n", i, np);
  });

  //'sigma', 'V', and the 'policy' control the layout of the SCS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  //Create the particle structure
  SellCSigma<Particle>* scs = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
                                                       ptcls_per_elem, element_gids);

  //Set initial and target positions so search will
  // find the parent elements
  setInitialPtclCoords(mesh, scs, output);
  setPtclIds(scs);
  setTargetPtclCoords(scs);

  //run search to move the particles to their starting elements
  search(mesh,scs, output);

  //define parameters controlling particle motion
  //move the particles in +y direction by 1/20th of the
  //pisces model's height
  fp_t heightOfDomain = 1.0;
  fp_t distance = heightOfDomain/20;
  fp_t dx = -0.5;
  fp_t dy = 0.8;
  fp_t dz = 0;

  if (comm_rank == 0)
    fprintf(stderr, "push distance %.3f push direction %.3f %.3f %.3f\n",
            distance, dx, dy, dz);

  o::LOs elmTags(ne, -1, "elmTagVals");
  mesh.add_tag(o::REGION, "has_particles", 1, elmTags);
  mesh.add_tag(o::VERT, "avg_density", 1, o::Reals(mesh.nverts(), 0));
  tagParentElements(mesh, scs, 0);
  render(mesh,0, comm_rank);

  Kokkos::Timer timer;
  Kokkos::Timer fullTimer;
  int iter;
  for(iter=1; iter<=NUM_ITERATIONS; iter++) {
    if(scs->num_ptcls == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if (comm_rank == 0)
      fprintf(stderr, "iter %d\n", iter);
    computeAvgPtclDensity(mesh, scs);
    timer.reset();
    push(scs, scs->num_ptcls, distance, dx, dy, dz);
    if (comm_rank == 0)
      fprintf(stderr, "push and transfer (seconds) %f\n", timer.seconds());
    if (output)
      writeDispVectors(scs);
    timer.reset();
    search(mesh,scs, output);
    if (comm_rank == 0)
      fprintf(stderr, "search, rebuild, and transfer (seconds) %f\n", timer.seconds());
    if(scs->num_ptcls == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    tagParentElements(mesh,scs,iter);
    render(mesh,iter, comm_rank);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (comm_rank == 0)
    fprintf(stderr, "%d iterations of pseudopush (seconds) %f\n", iter, fullTimer.seconds());
  
  //cleanup
  delete scs;

  Omega_h::vtk::write_parallel("pseudoPush_tf", &mesh, mesh.dim());
  fprintf(stderr, "done\n");
  return 0;
}
