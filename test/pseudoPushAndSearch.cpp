#include <Omega_h_mesh.hpp>
#include <Omega_h_bbox.hpp>
#include "pumipic_kktypes.hpp"
#include "pumipic_adjacency.hpp"
#include <particle_structs.hpp>
#include <Kokkos_Core.hpp>
#include "pumipic_mesh.hpp"
#include <fstream>
#define NUM_ITERATIONS 30

using particle_structs::lid_t;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using pumipic::fp_t;
using pumipic::Vector3d;

namespace o = Omega_h;
namespace p = pumipic;
namespace ps = particle_structs;

//To demonstrate push and adjacency search we store:
//-two fp_t[3] arrays, 'Vector3d', for the current and
// computed (pre adjacency search) positions, and
//-an integer to store the particles id
typedef MemberTypes<Vector3d, Vector3d, int> Particle;
typedef ps::ParticleStructure<Particle> PS;

void render(p::Mesh& picparts, int iter, int comm_rank) {
  std::stringstream ss;
  ss << "pseudoPush_t" << iter<<"_r"<<comm_rank;
  std::string s = ss.str();
  Omega_h::vtk::write_parallel(s, picparts.mesh(), picparts.dim());
}

void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

void writeDispVectors(PS* ptcls) {
  const int capacity = ptcls->capacity();
  auto x_nm1 = ptcls->get<0>();
  auto x_nm0 = ptcls->get<1>();
  auto pid_d = ptcls->get<2>();
  o::Write<o::Real> px_nm1(capacity*3);
  o::Write<o::Real> px_nm0(capacity*3);
  o::Write<o::LO> pid_w(capacity);

  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    pid_w[pid] = -1;
    if(mask > 0) {
      pid_w[pid] = pid_d(pid);
      for(int i=0; i<3; i++) {
        px_nm1[pid*3+i] = x_nm1(pid,i);
        px_nm0[pid*3+i] = x_nm0(pid,i);
      }
    }
  };
  ps::parallel_for(ptcls, lamb);

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

void push(PS* ptcls, int np, fp_t distance,
    fp_t dx, fp_t dy, fp_t dz) {
  Kokkos::Timer timer;
  auto position_d = ptcls->get<0>();
  auto new_position_d = ptcls->get<1>();

  const auto capacity = ptcls->capacity();

  fp_t disp[4] = {distance,dx,dy,dz};
  p::kkFpView disp_d("direction_d", 4);
  p::hostToDeviceFp(disp_d, disp);
  fprintf(stderr, "kokkos ps host to device transfer (seconds) %f\n", timer.seconds());

  o::Write<o::Real> ptclUnique_d(capacity, 0);

  double totTime = 0;
  timer.reset();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask) {
      fp_t dir[3];
      dir[0] = disp_d(0)*disp_d(1);
      dir[1] = disp_d(0)*disp_d(2);
      dir[2] = disp_d(0)*disp_d(3);
      new_position_d(pid,0) = position_d(pid,0) + dir[0] + ptclUnique_d[pid];
      new_position_d(pid,1) = position_d(pid,1) + dir[1] + ptclUnique_d[pid];
      new_position_d(pid,2) = position_d(pid,2) + dir[2] + ptclUnique_d[pid];
    }
  };
  ps::parallel_for(ptcls, lamb);

  totTime += timer.seconds();
  printTiming("ps push", totTime);
}

void tagParentElements(p::Mesh& picparts, PS* ptcls, int loop) {
  o::Mesh* mesh = picparts.mesh();
  //read from the tag
  o::LOs ehp_nm1 = mesh->get_array<o::LO>(picparts.dim(), "has_particles");
  o::Write<o::LO> ehp_nm0(ehp_nm1.size());
  auto set_ehp = OMEGA_H_LAMBDA(o::LO i) {
    ehp_nm0[i] = ehp_nm1[i];
  };
  o::parallel_for(ehp_nm1.size(), set_ehp, "set_ehp");

  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void) pid;
    if(mask > 0)
      ehp_nm0[e] = loop;
  };
  ps::parallel_for(ptcls, lamb);

  o::LOs ehp_nm0_r(ehp_nm0);
  mesh->set_tag(o::REGION, "has_particles", ehp_nm0_r);
}

void updatePtclPositions(PS* ptcls) {
  auto x_ps_d = ptcls->get<0>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto updatePtclPos = PS_LAMBDA(const int&, const int& pid, const bool&) {
    x_ps_d(pid,0) = xtgt_ps_d(pid,0);
    x_ps_d(pid,1) = xtgt_ps_d(pid,1);
    x_ps_d(pid,2) = xtgt_ps_d(pid,2);
    xtgt_ps_d(pid,0) = 0;
    xtgt_ps_d(pid,1) = 0;
    xtgt_ps_d(pid,2) = 0;
  };
  ps::parallel_for(ptcls, updatePtclPos);
}

void rebuild(p::Mesh& picparts, PS* ptcls, o::LOs elem_ids, const bool output) {
  updatePtclPositions(ptcls);
  const int ps_capacity = ptcls->capacity();
  auto ids = ptcls->get<2>();
  auto printElmIds = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(output && mask > 0)
      printf("elem_ids[%d] %d ptcl_id:%d\n", pid, elem_ids[pid], ids(pid));
  };
  ps::parallel_for(ptcls, printElmIds);

  PS::kkLidView ps_elem_ids("ps_elem_ids", ps_capacity);
  PS::kkLidView ps_process_ids("ps_process_ids", ps_capacity);
  Omega_h::LOs is_safe = picparts.safeTag();
  Omega_h::LOs elm_owners = picparts.entOwners(picparts.dim());
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      int new_elem = elem_ids[pid];
      ps_elem_ids(pid) = new_elem;
      ps_process_ids(pid) = comm_rank;
      if (new_elem != -1 && is_safe[new_elem] == 0) {
        ps_process_ids(pid) = elm_owners[new_elem];
      }
    }
  };
  ps::parallel_for(ptcls, lamb);

  ptcls->migrate(ps_elem_ids, ps_process_ids);

  printf("PS on rank %d has Elements: %d. Ptcls %d. Capacity %d. Rows %d.\n"
         , comm_rank, ptcls->nElems(), ptcls->nPtcls(), ptcls->capacity(), ptcls->numRows());
  ids = ptcls->get<2>();
  if (output) {
    auto printElms = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0)
        printf("Rank %d Ptcl: %d has Element %d and id %d\n", comm_rank, pid, e, ids(pid));
    };
    ps::parallel_for(ptcls, printElms);
  }
}

void search(p::Mesh& picparts, PS* ptcls, bool output) {
  o::Mesh* mesh = picparts.mesh();
  assert(ptcls->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 100;
  const auto psCapacity = ptcls->capacity();
  o::Write<o::LO> elem_ids(psCapacity,-1);
  Kokkos::Timer timer;
  auto x = ptcls->get<0>();
  auto xtgt = ptcls->get<1>();
  auto pid = ptcls->get<2>();
  o::Write<o::Real> xpoints_d(3 * psCapacity, "intersection points");
  o::Write<o::LO> xface_id(psCapacity, "intersection faces");
  bool isFound = p::search_mesh<Particle>(*mesh, ptcls, x, xtgt, pid, elem_ids,
                                          xpoints_d, xface_id, maxLoops);
  fprintf(stderr, "search_mesh (seconds) %f\n", timer.seconds());
  assert(isFound);
  //rebuild the PS to set the new element-to-particle lists
  timer.reset();
  rebuild(picparts, ptcls, elem_ids, output);
  fprintf(stderr, "rebuild (seconds) %f\n", timer.seconds());

}

//HACK to avoid having an unguarded comma in the PS PARALLEL macro
OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors(o::Reals const& a, o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

void setPtclIds(PS* ptcls) {
  auto pid_d = ptcls->get<2>();
  auto setIDs = PS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    pid_d(pid) = pid;
  };
  ps::parallel_for(ptcls, setIDs);
}

int setSourceElements(p::Mesh& picparts, PS::kkLidView ppe,
    const int mdlFace, const int numPtcls) {
  const auto elm_dim = picparts.dim();
  const auto side_dim = elm_dim-1;
  o::Mesh* mesh = picparts.mesh();
  auto face_class_ids = mesh->get_array<o::ClassId>(side_dim, "class_id");
  auto exposed_faces = o::mark_exposed_sides(mesh);
  o::Write<o::Byte> isClassOnFace(face_class_ids.size());
  o::parallel_for(face_class_ids.size(), OMEGA_H_LAMBDA(const int i) {
    isClassOnFace[i] = 0;
    if( face_class_ids[i] == mdlFace && exposed_faces[i] )
      isClassOnFace[i] = 1;
  });
  auto markedElms = mark_up(mesh, side_dim, elm_dim, isClassOnFace);
  o::Write<o::LO> markedElmIds(markedElms.size(),-1);
  o::parallel_for(markedElms.size(), OMEGA_H_LAMBDA(const int i) {
     markedElmIds[i] = markedElms[i] * i;
  });
  Omega_h::LO lastMarkedElm = o::get_max(o::LOs(markedElmIds));
  auto numMarkedElms = 0;
  Kokkos::parallel_reduce(markedElms.size(), OMEGA_H_LAMBDA(const int i, Omega_h::LO& lsum) {
    lsum += markedElms[i] > 0 ;
  }, numMarkedElms);
  printf("mesh elements classified on model face %d: %d\n", mdlFace, numMarkedElms);
  if (numMarkedElms > 0) {
    auto numPpe = numPtcls / numMarkedElms;
    printf("num ptcls per elm %d\n", numPpe);
    auto numPpeR = numPtcls % numMarkedElms;
    auto cells2nodes = mesh->get_adj(o::REGION, o::VERT).ab2b;
    auto nodes2coords = mesh->coords();
    o::parallel_for(markedElms.size(), OMEGA_H_LAMBDA(const int i) {
        auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(i));
        auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
        auto center = average(cell_nodes2coords);
        if( markedElms[i] )
          ppe[i] = numPpe + ( (i==lastMarkedElm) * numPpeR );
      });
    Omega_h::LO totPtcls = 0;
    Kokkos::parallel_reduce(ppe.size(), OMEGA_H_LAMBDA(const int i, Omega_h::LO& lsum) {
        lsum += ppe[i];
      }, totPtcls);
    assert(totPtcls == numPtcls);
    return totPtcls;
  }
  return 0;
}

void setInitialPtclCoords(p::Mesh& picparts, PS* ptcls, bool output) {
  //get centroid of parent element and set the child particle coordinates
  //most of this is copied from Omega_h_overlay.cpp get_cell_center_location
  //It isn't clear why the template parameter for gather_[verts|vectors] was
  //sized eight... maybe something to do with the 'Overlay'.  Given that there
  //are four vertices bounding a tet, I'm setting that parameter to four below.
  o::Mesh* mesh = picparts.mesh();
  auto cells2nodes = mesh->get_adj(o::REGION, o::VERT).ab2b;
  auto nodes2coords = mesh->coords();
  //set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
    auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
    auto center = average(cell_nodes2coords);
    if(mask > 0) {
      if (output)
        printf("elm %d xyz %f %f %f\n", e, center[0], center[1], center[2]);
      for(int i=0; i<3; i++)
        x_ps_d(pid,i) = center[i];
    }
  };
  ps::parallel_for(ptcls, lamb);
}

//Sunflower algorithm adapted from: https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
void setSunflowerPositions(PS* ptcls, const fp_t insetFaceDiameter, const fp_t insetFacePlane,
                           const fp_t insetFaceRim, const fp_t insetFaceCenter) {
  const fp_t insetFaceRadius = insetFaceDiameter/2;
  auto xtgt_ps_d = ptcls->get<1>();
  const o::LO n = ptcls->capacity();
  const fp_t phi = (sqrt(5) + 1) / 2;
  auto setPoints = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    const fp_t r = sqrt(pid + 0.5) / sqrt(n - 1 / 2);
    const fp_t theta = 2 * M_PI * pid / (phi*phi);
    xtgt_ps_d(pid, 0) = insetFaceCenter + insetFaceRadius * r * cos(theta);
    xtgt_ps_d(pid, 1) = insetFacePlane;
    xtgt_ps_d(pid, 2) = insetFaceCenter + insetFaceRadius * r * sin(theta);
  };
  ps::parallel_for(ptcls, setPoints);
}
void setLinearPositions(PS* ptcls, const fp_t insetFaceDiameter, const fp_t insetFacePlane,
                        const fp_t insetFaceRim, const fp_t insetFaceCenter){
  auto xtgt_ps_d = ptcls->get<1>();
  fp_t x_delta = insetFaceDiameter / (ptcls->capacity()-1);
  printf("x_delta %.4f\n", x_delta);
  if( ptcls->nPtcls() == 1 )
    x_delta = 0;
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      xtgt_ps_d(pid,0) = insetFaceCenter;
      xtgt_ps_d(pid,1) = insetFacePlane;
      xtgt_ps_d(pid,2) = insetFaceRim + (x_delta * pid);
    }
  };
  ps::parallel_for(ptcls, lamb);
}
void setTargetPtclCoords(PS* ptcls) {
  const fp_t insetFaceDiameter = 0.5;
  const fp_t insetFacePlane = 0.201; // just above the inset bottom face
  const fp_t insetFaceRim = -0.25; // in x
  const fp_t insetFaceCenter = 0; // in x and z
  setSunflowerPositions(ptcls, insetFaceDiameter, insetFacePlane, insetFaceRim, insetFaceCenter);
}

void computeAvgPtclDensity(p::Mesh& picparts, PS* ptcls){
  o::Mesh* mesh = picparts.mesh();
  //create an array to store the number of particles in each element
  o::Write<o::LO> elmPtclCnt_w(mesh->nelems(),0);
  //parallel loop over elements and particles
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {

    Kokkos::atomic_fetch_add(&(elmPtclCnt_w[e]), 1);
  };
  ps::parallel_for(ptcls, lamb);
  o::Write<o::Real> epc_w(mesh->nelems(),0);
  const auto convert = OMEGA_H_LAMBDA(o::LO i) {
     epc_w[i] = static_cast<o::Real>(elmPtclCnt_w[i]);
   };
  o::parallel_for(mesh->nelems(), convert, "convert_to_real");
  o::Reals epc(epc_w);
  mesh->add_tag(o::REGION, "element_particle_count", 1, o::Reals(epc));
  //get the list of elements adjacent to each vertex
  auto verts2elems = mesh->ask_up(o::VERT, picparts.dim());
  //create a device writeable array to store the computed density
  o::Write<o::Real> ad_w(mesh->nverts(),0);
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
  o::parallel_for(mesh->nverts(), accumulate, "calculate_avg_density");
  o::Read<o::Real> ad_r(ad_w);
  mesh->set_tag(o::VERT, "avg_density", ad_r);
}

o::Mesh readMesh(char* meshFile, o::Library& lib) {
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self());
  } else {
    std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  const int numargs = 8;
  if( argc != numargs ) {
    auto args = " <mesh> <owner_file> <numPtcls> "
      "<initial model face> <push vector>";
    std::cout << "Usage: " << argv[0] << args << "\n";
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
  auto full_mesh = readMesh(argv[1], lib);
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
  p::Mesh picparts(full_mesh,owner);
  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts(); //caching adjacency info

  if (comm_rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh->nverts(), mesh->nedges(),
           mesh->nfaces(), mesh->nelems());

  /* Particle data */
  int numPtcls = atoi(argv[3]);
  const bool output = numPtcls <= 30;
  if (comm_rank != 0)
    numPtcls = 0;
  Omega_h::Int ne = mesh->nelems();
  if (comm_rank == 0)
    fprintf(stderr, "number of elements %d number of particles %d\n",
            ne, numPtcls);
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = mesh_element_gids[i];
  });
  const int mdlFace = atoi(argv[4]);
  int actualParticles = setSourceElements(picparts,ptcls_per_elem,mdlFace,numPtcls);
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
    if (output && np > 0)
     printf("ppe[%d] %d\n", i, np);
  });

  //'sigma', 'V', and the 'policy' control the layout of the PS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  //Create the particle structure
  PS* ptcls = new SellCSigma<Particle>(policy, sigma, V, ne, actualParticles,
                                       ptcls_per_elem, element_gids);
  setInitialPtclCoords(picparts, ptcls, output);
  setPtclIds(ptcls);

  //define parameters controlling particle motion
  //move the particles in +y direction by 1/20th of the
  //pisces model's height
  const double pushDir[3] = {atof(argv[5]), atof(argv[6]), atof(argv[7])};
  auto bb = o::get_bounding_box<3>(&full_mesh);
  double maxDimLen = 0;
  printf("bbox ");
  for(int i=0; i< picparts.dim(); i++) {
    printf("%3d %.3f %.3f ", i, bb.min[i], bb.max[i]);
    auto len = bb.max[i]-bb.min[i];
    if( len > maxDimLen ) maxDimLen = len;
  }
  printf("\n");
  const fp_t heightOfDomain = maxDimLen;
  const fp_t distance = heightOfDomain/20;
  const fp_t dx = pushDir[0];
  const fp_t dy = pushDir[1];
  const fp_t dz = pushDir[2];

  if (comm_rank == 0)
    fprintf(stderr, "push distance %.3f push direction %.3f %.3f %.3f\n",
            distance, dx, dy, dz);

  o::LOs elmTags(ne, -1, "elmTagVals");
  mesh->add_tag(o::REGION, "has_particles", 1, elmTags);
  mesh->add_tag(o::VERT, "avg_density", 1, o::Reals(mesh->nverts(), 0));
  tagParentElements(picparts, ptcls, 0);
  render(picparts,0, comm_rank);

  Kokkos::Timer timer;
  Kokkos::Timer fullTimer;
  int iter;
  int np;
  int ps_np;
  for(iter=1; iter<=NUM_ITERATIONS; iter++) {
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if (comm_rank == 0)
      fprintf(stderr, "iter %d\n", iter);
    //computeAvgPtclDensity(picparts, ptcls);
    timer.reset();
    push(ptcls, ptcls->nPtcls(), distance, dx, dy, dz);
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank == 0)
      fprintf(stderr, "push and transfer (seconds) %f\n", timer.seconds());
    if (output)
      writeDispVectors(ptcls);
    timer.reset();
    search(picparts,ptcls, output);
    if (comm_rank == 0)
      fprintf(stderr, "search, rebuild, and transfer (seconds) %f\n", timer.seconds());
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    tagParentElements(picparts,ptcls,iter);
    render(picparts,iter, comm_rank);
  }
  if (comm_rank == 0)
    fprintf(stderr, "%d iterations of pseudopush (seconds) %f\n", iter, fullTimer.seconds());

  //cleanup
  delete ptcls;

  Omega_h::vtk::write_parallel("pseudoPush_tf", mesh, picparts.dim());
  if (!comm_rank)
    fprintf(stderr, "done\n");
  return 0;
}
