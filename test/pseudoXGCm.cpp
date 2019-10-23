#include <Omega_h_mesh.hpp>
#include <SCS_Macros.h>
#include <SellCSigma.h>
#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_profiling.hpp"
#include "pseudoXGCmTypes.hpp"
#include "gyroScatter.hpp"
#include <fstream>
#include "ellipticalPush.hpp"
#include <random>
#define ELEMENT_SEED 1024*1024
#define PARTICLE_SEED 512*512

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

void render(p::Mesh& picparts, int iter, int comm_rank) {
  std::stringstream ss;
  ss << "pseudoPush_r" << comm_rank<<"_t"<<iter;
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

void tagParentElements(p::Mesh& picparts, SCS* scs, int loop) {
  o::Mesh* mesh = picparts.mesh();
  //read from the tag
  o::LOs ehp_nm1 = mesh->get_array<o::LO>(picparts.dim(), "has_particles");
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

void updatePtclPositions(SCS* scs) {
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

void rebuild(p::Mesh& picparts, SCS* scs, o::LOs elem_ids, const bool output) {
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto ids = scs->get<2>();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(output && mask > 0)
      printf("elem_ids[%d] %d ptcl_id:%d\n", pid, elem_ids[pid], ids(pid));
  };
  scs->parallel_for(printElmIds);

  SCS::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);
  SCS::kkLidView scs_process_ids("scs_process_ids", scs_capacity);
  Omega_h::LOs is_safe = picparts.safeTag();
  Omega_h::LOs elm_owners = picparts.entOwners(picparts.dim());
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

void search(p::Mesh& picparts, SCS* scs, bool output) {
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  o::Mesh* mesh = picparts.mesh();
  assert(scs->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 200;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  auto x = scs->get<0>();
  auto xtgt = scs->get<1>();
  auto pid = scs->get<2>();
  bool isFound = p::search_mesh_2d<Particle>(*mesh, scs, x, xtgt, pid, elem_ids, maxLoops);
  assert(isFound);
  //rebuild the SCS to set the new element-to-particle lists
  rebuild(picparts, scs, elem_ids, output);
}

void setPtclIds(SCS* scs) {
  auto pid_d = scs->get<2>();
  auto setIDs = SCS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    pid_d(pid) = pid;
  };
  scs->parallel_for(setIDs);
}

int setSourceElements(p::Mesh& picparts, SCS::kkLidView ppe,
    const int mdlFace, const int numPtclsPerRank) {
  //Deterministically generate random number of particles on each element with classification less than mdlFace
  int comm_rank = picparts.comm()->rank();
  const auto elm_dim = picparts.dim();
  o::Mesh* mesh = picparts.mesh();
  auto face_class_ids = mesh->get_array<o::ClassId>(elm_dim, "class_id");
  auto face_owners = picparts.entOwners(elm_dim);
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

void setInitialPtclCoords(p::Mesh& picparts, SCS* scs, bool output) {
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
  o::Mesh* mesh = picparts.mesh();
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
  if( scs->nPtcls() == 1 )
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

o::Mesh readMesh(char* meshFile, o::Library& lib) {
  const auto rank = lib.world()->rank();
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    if(!rank)
      std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    if(!rank)
      std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self(), true);
  } else {
    if(!rank)
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
  const int numargs = 10;
  if( argc != numargs ) {
    printf("numargs %d expected %d\n", argc, numargs);
    auto args = " <mesh> <owner_file> <numPtcls> "
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
  auto full_mesh = readMesh(argv[1], lib);

  MPI_Barrier(MPI_COMM_WORLD);
 
  if(!comm_rank)
    printf("Mesh loaded with <v e f> %d %d %d\n",full_mesh.nverts(),full_mesh.nedges(),
        full_mesh.nfaces());

  const auto vtx_to_elm = full_mesh.ask_up(0, 2);
  const auto edge_to_elm = full_mesh.ask_up(1, 2);

  if(!comm_rank) {
    fprintf(stderr, "done mesh topo checks\n");
    fprintf(stderr, "partition file %s\n", argv[2]);
    fprintf(stderr, "input buffer method %s safe method %s\n", argv[6], argv[7]);
  }

  const auto bufferMethod = pumipic::Input::getMethod(argv[6]);
  const auto safeMethod = pumipic::Input::getMethod(argv[7]);
  assert(bufferMethod>=0);
  assert(safeMethod>=0);
  //Create picparts using classification with the full mesh buffered and minimum safe zone
  p::Input input(full_mesh, argv[2], bufferMethod, safeMethod);
  if(!comm_rank)
    input.printInfo();
  MPI_Barrier(MPI_COMM_WORLD);
  p::Mesh picparts(input);
  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts(); //caching adjacency info
 
  //Build gyro avg mappings
  const auto rmax = 0.038;
  const auto numRings = 3;
  const auto ptsPerRing = 8;
  const auto theta = 0.0;
  setGyroConfig(rmax,numRings,ptsPerRing,theta);
  if (!comm_rank) printGyroConfig();
  Omega_h::LOs forward_map;
  Omega_h::LOs backward_map;
  createGyroRingMappings(mesh, forward_map, backward_map);
  
  /* Particle data */
  const long int numPtcls = atol(argv[3]);
  const int numPtclsPerRank = numPtcls / comm_size;
  const bool output = numPtclsPerRank <= 30;

  long int totNumReqPtcls = 0;
  const long int numPtclsPerRank_li = numPtclsPerRank;
  MPI_Allreduce(&numPtclsPerRank_li, &totNumReqPtcls, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (!comm_rank)
    fprintf(stderr, "particles requested %ld %ld\n", numPtcls, totNumReqPtcls);

  Omega_h::Int ne = mesh->nelems();

  
  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  SCS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = mesh_element_gids[i];
  });
  const int mdlFace = atoi(argv[4]);
  int actualParticles = setSourceElements(picparts,ptcls_per_elem,mdlFace,numPtclsPerRank);
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
  //Create the particle structure
  SellCSigma<Particle>* scs = new SellCSigma<Particle>(policy, sigma, V, ne, actualParticles,
                                                       ptcls_per_elem, element_gids);
  setInitialPtclCoords(picparts, scs, output);
  setPtclIds(scs);

  //define parameters controlling particle motion
  const double h = 1.72479370-.08;
  const auto k = .020558260;
  const auto d = 0.6;
  ellipticalPush::setup(scs, h, k, d);
  const auto degPerPush = atof(argv[8]);
  if (!comm_rank)
    fprintf(stderr, "degrees per elliptical push %f\n", degPerPush);

  if (comm_rank == 0)
    fprintf(stderr, "ellipse center %f %f ellipse ratio %.3f\n", h, k, d);

  o::LOs elmTags(ne, -1, "elmTagVals");
  mesh->add_tag(o::FACE, "has_particles", 1, elmTags);
  mesh->add_tag(o::VERT, "avg_density", 1, o::Reals(mesh->nverts(), 0));
  const auto fwdTagName = "ptclToMeshScatterFwd";
  mesh->add_tag(o::VERT, fwdTagName, 1, o::Reals(mesh->nverts(), 0));
  const auto bkwdTagName = "ptclToMeshScatterBkwd";
  mesh->add_tag(o::VERT, bkwdTagName, 1, o::Reals(mesh->nverts(), 0));
  const auto syncTagName = "ptclToMeshSync";
  mesh->add_tag(o::VERT, syncTagName, 2, o::Reals(mesh->nverts()*2, 0));
  tagParentElements(picparts, scs, 0);

  const auto enable_prebarrier = atoi(argv[9]);
  if(enable_prebarrier) {
    if(!comm_rank)
      fprintf(stderr, "pre-barrier enabled\n");
    particle_structs::enable_prebarrier();
    pumipic_enable_prebarrier();
  }
  Kokkos::Timer timer;
  Kokkos::Timer fullTimer;
  int iter;
  long int totNp;
  long int scs_np;
  for(iter=1; iter<=maxIter; iter++) {
    if(!comm_rank || (comm_rank == comm_size/2))
      scs->printMetrics();
    scs_np = scs->nPtcls();
    MPI_Allreduce(&scs_np, &totNp, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(totNp == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if (!comm_rank)
      fprintf(stderr, "iter %d particles %ld\n", iter, totNp);
    getMemImbalance(scs_np!=0);
    timer.reset();
    ellipticalPush::push(scs, *mesh, degPerPush, iter);
    MPI_Barrier(MPI_COMM_WORLD);
    timer.reset();
    search(picparts,scs, output);
    scs_np = scs->nPtcls();
    MPI_Allreduce(&scs_np, &totNp, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(totNp == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    tagParentElements(picparts,scs,iter);
    if(output && !(iter%100))
      render(picparts,iter, comm_rank);
    gyroScatter(mesh,scs,forward_map,fwdTagName);
    gyroScatter(mesh,scs,backward_map,bkwdTagName);
    gyroSync(picparts,fwdTagName,bkwdTagName,syncTagName);
  }
  if (comm_rank == 0)
    fprintf(stderr, "%d iterations of pseudopush (seconds) %f\n", iter, fullTimer.seconds());
  
  //cleanup
  delete scs;

  if (!comm_rank)
    fprintf(stderr, "done\n");
  return 0;
}
