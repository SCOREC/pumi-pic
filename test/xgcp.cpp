#include <fstream>
#include <random>
#include <chrono>
#include <thread>
#include <xgcp_mesh.hpp>
#include <xgcp_push.hpp>
#include <xgcp_gyro_scatter.hpp>
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
void getPtclImbalance(int ptclCnt);
void printTimerResolution();
void setInitialPtclCoords(p::Mesh* picparts, SCS_I* scs, bool output);
int setSourceElements(p::Mesh* picparts, SCS_I::kkLidView ppe,
                      const int mdlFace, const int numPtclsPerRank);
void setPtclIds(SCS_I* scs);
void search(xgcp::Mesh& picparts, SCS_I* scs, bool output);
void tagParentElements(xgcp::Mesh& mesh, SCS_I* scs, int loop);
void render(xgcp::Mesh& mesh, int iter, int comm_rank);

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
  const long int numPtcls = atol(argv[3]);
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
  const int mdlFace = atoi(argv[6]);
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

  const auto maxIter = atoi(argv[7]);
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

  //Setup push
  const double h = 1.72479370-.08;
  const auto k = .020558260;
  const auto d = 0.6;
  xgcp::ellipticalPush::setup(scs, h, k, d);
  const auto degPerPush = atof(argv[10]);
  if (!comm_rank)
    fprintf(stderr, "degrees per elliptical push %f\n", degPerPush);

  if (comm_rank == 0)
    fprintf(stderr, "ellipse center %f %f ellipse ratio %.3f\n", h, k, d);

  //Add tags to mesh
  o::LOs elmTags(ne, -1, "elmTagVals");
  omesh->add_tag(o::FACE, "has_particles", 1, elmTags);
  omesh->add_tag(o::VERT, "avg_density", 1, o::Reals(omesh->nverts(), 0));

  const auto enable_prebarrier = atoi(argv[11]);
  if(enable_prebarrier) {
    if(!comm_rank)
      fprintf(stderr, "pre-barrier enabled\n");
    particle_structs::enable_prebarrier();
    pumipic_enable_prebarrier();
  }

  //Main iteration loop
  Kokkos::Timer timer;
  Kokkos::Timer fullTimer;
  int iter;
  long int totNp;
  long int scs_np;
  for(iter=1; iter<=maxIter; iter++) {
    //Print Metrics on rank 0 and comm_size/2
    if(!comm_rank || (comm_rank == comm_size/2))
      scs->printMetrics();
    //Stop if there are no more particles
    scs_np = scs->nPtcls();
    MPI_Allreduce(&scs_np, &totNp, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(totNp == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    //Print Iteration stats
    if (!comm_rank)
      fprintf(stderr, "iter %d particles %ld\n", iter, totNp);
    getMemImbalance(scs_np!=0);
    getPtclImbalance(scs_np);
    timer.reset();
    //Push Particles
    xgcp::ellipticalPush::push(scs, *omesh, degPerPush, iter);
    MPI_Barrier(MPI_COMM_WORLD);
    //Perform search and rebuild
    search(mesh, scs, output);
    scs_np = scs->nPtcls();
    MPI_Allreduce(&scs_np, &totNp, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(totNp == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    tagParentElements(mesh,scs,iter);

    //Perform gyro scatter
    xgcp::gyroScatter(mesh, scs);
  }
  if (comm_rank == 0)
    fprintf(stderr, "%d iterations of pseudopush (seconds) %f\n", iter, fullTimer.seconds());
  mesh.applyGyroFieldsToTags();
  render(mesh, iter, comm_rank);

  //cleanup
  delete scs;

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

void getPtclImbalance(int ptclCnt) {
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  long ptcls=ptclCnt;
  long max=0;
  long tot=0;
  int hasptcls = (ptclCnt > 0);
  int rankswithptcls=0;
  MPI_Allreduce(&ptcls, &max, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&ptcls, &tot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&hasptcls, &rankswithptcls, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const double avg=static_cast<double>(tot)/rankswithptcls;
  const double imb=max/avg;
  if(!comm_rank) {
    printf("ranks with particles %d particle imbalance %f\n",
        rankswithptcls, imb);
  }
  if( ptcls == max ) {
    printf("%d peak particle count %ld, avg usage %f\n", comm_rank, max, avg);
  }
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

  double nppe = numPtclsPerRank / numMarked;
  int ne = mesh->nelems();
  o::HostWrite<o::LO> rand_per_elem(ne);
  for (int i = 0; i < ne; ++i)
    rand_per_elem[i] = 0;

  int r;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  r+= 1;
  int total = 0;
  std::default_random_engine generator(ELEMENT_SEED * r * r ^ (12345 * r));
  Omega_h::HostRead<o::LO> isFaceOnClass_host(isFaceOnClass);

  if (nppe < 5) {
    //Gaussian Random generator to randomly select an element to add each particle to
    std::normal_distribution<double> dist(ne/2.0, ne / 4.0);

    while (total < numPtclsPerRank) {
      int elem = round(dist(generator));
      if (elem < 0 || elem >= ne)
        continue;
      if (!isFaceOnClass_host[elem])
        continue;
      ++rand_per_elem[elem];
      ++total;
    }
  }
  else {
    //Gaussian Random generator with mean = number of particles per element
    std::normal_distribution<double> dist(nppe, nppe / 4);
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
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  comm_rank++;
  o::HostWrite<o::Real> rand_num_per_ptcl(2*scs->capacity());
  std::default_random_engine generator(PARTICLE_SEED / comm_rank);
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

void updatePtclPositions(SCS_I* scs) {
  auto x_scs_d = scs->get<0>();
  auto xtgt_scs_d = scs->get<1>();
  auto updatePtclPos = SCS_LAMBDA(const int&, const int& pid, const bool&) {
    x_scs_d(pid,0) = xtgt_scs_d(pid,0);
    x_scs_d(pid,1) = xtgt_scs_d(pid,1);
    x_scs_d(pid,2) = xtgt_scs_d(pid,2);
    xtgt_scs_d(pid,0) = 0;
    xtgt_scs_d(pid,1) = 0;
    xtgt_scs_d(pid,2) = 0;
  };
  scs->parallel_for(updatePtclPos);
}

void rebuild(xgcp::Mesh& mesh, SCS_I* scs, o::LOs elem_ids, const bool output) {
  p::Mesh* picparts = mesh.pumipicMesh();
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto ids = scs->get<2>();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(output && mask > 0)
      printf("elem_ids[%d] %d ptcl_id:%d\n", pid, elem_ids[pid], ids(pid));
  };
  scs->parallel_for(printElmIds);

  SCS_I::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);
  SCS_I::kkLidView scs_process_ids("scs_process_ids", scs_capacity);
  Omega_h::LOs is_safe = picparts->safeTag();
  Omega_h::LOs elm_owners = picparts->entOwners(picparts->dim());
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      int new_elem = elem_ids[pid];
      scs_elem_ids(pid) = new_elem;
      scs_process_ids(pid) = comm_rank;
      if (new_elem != -1 && is_safe[new_elem] == 0) {
        scs_process_ids(pid) = elm_owners[new_elem];
      }
    }
  };
  scs->parallel_for(lamb);

  scs->migrate(scs_elem_ids, scs_process_ids);

  ids = scs->get<2>();
  if (output) {
    auto printElms = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0)
        printf("Rank %d Ptcl: %d has Element %d and id %d\n", comm_rank, pid, e, ids(pid));
    };
    scs->parallel_for(printElms);
  }
}

void search(xgcp::Mesh& mesh, SCS_I* scs, bool output) {
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  o::Mesh* omesh = mesh.omegaMesh();
  assert(scs->nElems() == omesh->nelems());
  Omega_h::LO maxLoops = 200;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  auto x = scs->get<0>();
  auto xtgt = scs->get<1>();
  auto pid = scs->get<2>();
  bool isFound = p::search_mesh_2d<xgcp::Ion>(*omesh, scs, x, xtgt, pid, elem_ids, maxLoops);
  assert(isFound);
  //rebuild the SCS to set the new element-to-particle lists
  rebuild(mesh, scs, elem_ids, output);
}

void tagParentElements(xgcp::Mesh& mesh, SCS_I* scs, int loop) {
  //read from the tag
  o::LOs ehp_nm1 = mesh->get_array<o::LO>(mesh.dim(), "has_particles");
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
  mesh->set_tag(o::FACE, "has_particles", ehp_nm0_r);
}

void render(xgcp::Mesh& mesh, int iter, int comm_rank) {
  std::stringstream ss;
  ss << "pseudoPush_r" << comm_rank<<"_t"<<iter;
  std::string s = ss.str();
  Omega_h::vtk::write_parallel(s, mesh.omegaMesh(), mesh.dim());
}
