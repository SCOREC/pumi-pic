#include <fstream>
#include <random>
#include <chrono>
#include <thread>
#include <xgcp_mesh.hpp>
#include <SCS_Macros.h>
#include <SellCSigma.h>
#include <Omega_h_for.hpp>
#include "pumipic_adjacency.hpp"
#include "pumipic_profiling.hpp"

#define ELEMENT_SEED 1024*1024
#define PARTICLE_SEED 512*512

using xgcp::fp_t;
using xgcp::SCS_I;

namespace p = pumipic;
namespace ps = particle_structs;
namespace o = Omega_h;

void getMemImbalance(int hasptcls);
void printTimerResolution();
void setInitialPtclCoords(p::Mesh* picparts, SCS_I* scs, bool output);
int setSourceElements(p::Mesh* picparts, SCS_I::kkLidView ppe,
                      const int mdlFace, const int numPtclsPerRank);
void setPtclIds(SCS_I* scs);

int main(int argc, char* argv[]) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  const int numargs = 12;
  if( argc != numargs ) {
    printf("numargs %d expected %d\n", argc, numargs);
    auto args = " <mesh> <owner_file> <numPtcls> "
      "<num planes> <num procs per group>"
      "<max initial model face> <maxIterations> "
      "<buffer method=[bfs|full]> <safe method=[bfs|full]> "
      "<degrees per elliptical push>"
      "<enable prebarrier>";
    std::cout << "Usage: " << argv[0] << args << "\n";
    exit(1);
  }
  getMemImbalance(1);
  if (comm_rank == 0) {
    printf("world ranks %d\n", comm_size);
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

  char* mesh_file = argv[1];
  char* partition_file = argv[2];
  const auto bufferMethod = pumipic::Input::getMethod(argv[8]);
  const auto safeMethod = pumipic::Input::getMethod(argv[9]);
  assert(bufferMethod>=0);
  assert(safeMethod>=0);
  int num_planes = atoi(argv[4]);
  int num_processes_per_group = atoi(argv[5]);
  xgcp::Input input(lib, mesh_file, partition_file, num_planes, num_processes_per_group,
                    bufferMethod, safeMethod);
  xgcp::Mesh mesh(input);
  p::Mesh* picparts = mesh.pumipicMesh();
  o::Mesh* omesh = mesh.omegaMesh();

  /* Particle data */
  const long int numPtcls = 1000;//atol(argv[3]);
  const int numPtclsPerRank = numPtcls / comm_size;
  const bool output = numPtclsPerRank <= 30;

  long int totNumReqPtcls = 0;
  const long int numPtclsPerRank_li = numPtclsPerRank;
  MPI_Allreduce(&numPtclsPerRank_li, &totNumReqPtcls, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (!comm_rank)
    fprintf(stderr, "particles requested %ld %ld\n", numPtcls, totNumReqPtcls);

  Omega_h::Int ne = mesh.nelems();
  SCS_I::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  SCS_I::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts->globalIds(mesh.dim());
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = mesh_element_gids[i];
  });
  const int mdlFace = atoi(argv[4]);
  int actualParticles = setSourceElements(picparts, ptcls_per_elem, mdlFace, numPtclsPerRank);
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
    if (output && np > 0)
     printf("ppe[%d] %d\n", i, np);
  });

  long int totNumPtcls = 0;
  long int actualParticles_li = actualParticles;
  MPI_Allreduce(&actualParticles_li, &totNumPtcls, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (!comm_rank)
    fprintf(stderr, "particles created %ld\n", totNumPtcls);

  const auto maxIter = atoi(argv[5]);
  if (!comm_rank)
    fprintf(stderr, "max iterations: %d\n", maxIter);

  //'sigma', 'V', and the 'policy' control the layout of the SCS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  //Create the Ion particle structure
  SCS_I* scs = new SCS_I(policy, sigma, V, ne, actualParticles, ptcls_per_elem, element_gids);
  setInitialPtclCoords(picparts, scs, output);
  setPtclIds(scs);


  //cleanup

  if (!comm_rank)
    fprintf(stderr, "done\n");
  return 0;
}

void getMemImbalance(int hasptcls) {
#ifdef SCS_USE_CUDA
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  const long used=total-free;
  long maxused=0;
  long totused=0;
  int rankswithptcls=0;
  MPI_Allreduce(&used, &maxused, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&used, &totused, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&hasptcls, &rankswithptcls, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const double avg=static_cast<double>(totused)/rankswithptcls;
  const double imb=maxused/avg;
  if(!comm_rank) {
    printf("ranks with particles %d memory usage imbalance %f\n",
        rankswithptcls, imb);
  }
  if( used == maxused ) {
    printf("%d peak mem usage %ld, avg usage %f\n", comm_rank, maxused, avg);
  }
#endif
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

void setPtclIds(SCS_I* scs) {
  auto pid_d = scs->get<2>();
  auto setIDs = SCS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    pid_d(pid) = pid;
  };
  scs->parallel_for(setIDs);
}

int setSourceElements(p::Mesh* picparts, SCS_I::kkLidView ppe,
    const int mdlFace, const int numPtclsPerRank) {
  //Deterministically generate random number of particles on each element with classification less than mdlFace
  int comm_rank = picparts->comm()->rank();
  const auto elm_dim = picparts->dim();
  o::Mesh* mesh = picparts->mesh();
  auto face_class_ids = mesh->get_array<o::ClassId>(elm_dim, "class_id");
  auto face_owners = picparts->entOwners(elm_dim);
  o::Write<o::LO> isFaceOnClass(face_class_ids.size(), 0);
  o::parallel_for(face_class_ids.size(), OMEGA_H_LAMBDA(const int i) {
    if( face_class_ids[i] <= mdlFace && face_owners[i] == comm_rank)
      isFaceOnClass[i] = 1;
  });
  o::LO numMarked = o::get_sum(o::LOs(isFaceOnClass));
  if(!numMarked)
    return 0;

  int nppe = numPtclsPerRank / numMarked;
  o::HostWrite<o::LO> rand_per_elem(mesh->nelems());

  //Gaussian Random generator with mean = number of particles per element
  std::default_random_engine generator(ELEMENT_SEED);
  std::normal_distribution<double> dist(nppe, nppe / 4);

  Omega_h::HostWrite<o::LO> isFaceOnClass_host(isFaceOnClass);
  int total = 0;
  int last = -1;
  for (int i = 0; i < mesh->nelems(); ++i) {
    rand_per_elem[i] = 0;
    if (isFaceOnClass_host[i] && total < numPtclsPerRank ) {
      last = i;
      rand_per_elem[i] = round(dist(generator));
      if (rand_per_elem[i] < 0)
        rand_per_elem[i] = 0;
      total += rand_per_elem[i];
      //Stop if we hit the number of particles
      if (total > numPtclsPerRank) {
        int over = total - numPtclsPerRank;
        rand_per_elem[i] -= over;
      }
    }
  }
  //If we didn't put all particles in, fill them in the last element we touched
  if (total < numPtclsPerRank) {
    int under = numPtclsPerRank - total;
    rand_per_elem[last] += under;
  }
  o::Write<o::LO> ppe_write(rand_per_elem);


  int np = o::get_sum(o::LOs(ppe_write));
  o::parallel_for(mesh->nelems(), OMEGA_H_LAMBDA(const o::LO& i) {
    ppe(i) = ppe_write[i];
  });
  return np;
}

void setInitialPtclCoords(p::Mesh* picparts, SCS_I* scs, bool output) {
  //Randomly distrubite particles within each element (uniformly within the element)
  //Create a deterministic generation of random numbers on the host with 2 number per particle

  o::HostWrite<o::Real> rand_num_per_ptcl(2*scs->capacity());
  std::default_random_engine generator(PARTICLE_SEED);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < scs->capacity(); ++i) {
    o::Real x = dist(generator);
    o::Real y = dist(generator);
    if (x+y > 1) {
      x = 1-x;
      y = 1-y;
    }
    rand_num_per_ptcl[2*i] = x;
    rand_num_per_ptcl[2*i+1] = y;
  }
  o::Write<o::Real> rand_nums(rand_num_per_ptcl);
  o::Mesh* mesh = picparts->mesh();
  auto cells2nodes = mesh->get_adj(o::FACE, o::VERT).ab2b;
  auto nodes2coords = mesh->coords();
  //set particle positions and parent element ids
  auto x_scs_d = scs->get<0>();
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = o::gather_verts<3>(cells2nodes, o::LO(e));
      auto vtxCoords = o::gather_vectors<3,2>(nodes2coords, elmVerts);
      o::Real r1 = rand_nums[2*pid];
      o::Real r2 = rand_nums[2*pid+1];
      // X = A + r1(B-A) + r2(C-A)
      for (int i = 0; i < 2; i++)
        x_scs_d(pid,i) = vtxCoords[0][i] + r1 * (vtxCoords[1][i] - vtxCoords[0][i])
                                        + r2 * (vtxCoords[2][i] - vtxCoords[0][i]);
      x_scs_d(pid,2) = 0;
      if (output)
        printf("pid %d: %.3f %.3f %.3f\n", pid, x_scs_d(pid,0), x_scs_d(pid,1), x_scs_d(pid,2));
    }
  };
  scs->parallel_for(lamb);
}
