#include <fstream>

#include <particle_structs.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>
#include <pumipic_lb.hpp>


typedef pumipic::MemberTypes<int> Particle;
typedef pumipic::ParticleStructure<Particle> PS;

double printImb(PS* ptcls);
void balancePtcls(pumipic::Mesh& picparts, PS* ptcls, pumipic::ParticleBalancer& balancer);

int testBalanceArray(pumipic::Mesh& picparts, pumipic::ParticleBalancer& balancer);
int testBalancePS(pumipic::Mesh& picparts, pumipic::ParticleBalancer& balancer);

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 3) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename>\n", argv[0]);
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  //**********Load the mesh in serial everywhere*************//
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.self());
  int dim = mesh.dim();
  int ne = mesh.nents(dim);
  if (rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh.nverts(), mesh.nedges(),
           mesh.nfaces(), mesh.nelems());


  Omega_h::HostWrite<Omega_h::LO> host_owners(ne);
  if (comm_size > 1) {
    std::ifstream in_str(argv[2]);
    if (!in_str) {
      if (!rank)
        fprintf(stderr,"Cannot open file %s\n", argv[2]);
      return EXIT_FAILURE;
    }
    int own;
    int index = 0;
    while(in_str >> own)
      host_owners[index++] = own;
  }
  else
    for (int i = 0; i < mesh.nelems(); ++i)
      host_owners[i] = 0;

  //Owner of each element
  Omega_h::Write<Omega_h::LO> owner(host_owners);
  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::BFS,
                       pumipic::Input::FULL);
  pumipic::Mesh picparts(input);

  //Build Particle Balancer
  pumipic::ParticleBalancer balancer(picparts);

  int fails = 0;
  fails += testBalanceArray(picparts, balancer);
  fails += testBalancePS(picparts, balancer);

  if (!rank && fails == 0) {
    fprintf(stderr, "All Tests Passed\n");
  }
  return fails;
}

int testBalanceArray(pumipic::Mesh& picparts, pumipic::ParticleBalancer& balancer) {
  int fail = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (!rank)
    fprintf(stderr, "Starting test for balancing an array of particles per element\n");
  Omega_h::LO ne = picparts->nelems();
  Omega_h::LO num_ptcls = (rank+1) * 50*ne;
  Kokkos::View<Omega_h::LO*> ptcls_per_elem("ptcls_per_elem", picparts->nelems());
  Omega_h::parallel_for(picparts->nelems(), OMEGA_H_LAMBDA(const int& i) {
    ptcls_per_elem(i) = num_ptcls/ne;
  });

  Omega_h::LO sum_start, max_start;
  MPI_Allreduce(&num_ptcls, &sum_start, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&num_ptcls, &max_start, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (!rank) {
    Omega_h::Real avg = sum_start / picparts.comm()->size();
    Omega_h::Real imb = max_start / avg;
    fprintf(stderr, "Imbalance at start is %f\n", imb);
  }

  auto new_procs = balancer.partition(picparts, ptcls_per_elem, 1.05);

  Omega_h::Write<Omega_h::LO> send_ptcls(picparts.comm()->size(), 0);
  auto countSendingPtcls = OMEGA_H_LAMBDA(const Omega_h::LO ptcl) {
    Kokkos::atomic_add(&(send_ptcls[new_procs[ptcl]]),1);
  };
  Omega_h::parallel_for(new_procs.size(), countSendingPtcls, "countSendingPtcls");

  Omega_h::HostWrite<Omega_h::LO> send_ptcls_host(send_ptcls);
  Omega_h::LO* ptcls_per_rank = new Omega_h::LO[send_ptcls_host.size()];
  MPI_Allreduce(send_ptcls_host.data(), ptcls_per_rank, picparts.comm()->size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (!rank) {
    Omega_h::LO total = 0;
    Omega_h::LO max = 0;
    for (int i = 0; i < picparts.comm()->size(); ++i) {
      int val = ptcls_per_rank[i];
      total += val;
      if (val > max)
        max = val;
    }
    Omega_h::Real avg = total * 1.0/ picparts.comm()->size();
    Omega_h::Real imb = max/avg;
    fprintf(stderr, "Imbalance after balancing is %f\n\n", imb);

    if (imb > 1.3)
      fail = 1;
  }
  delete [] ptcls_per_rank;
  return fail;
}
int testBalancePS(pumipic::Mesh& picparts, pumipic::ParticleBalancer& balancer) {
  int fail = 0;
  //Create 100 particles/elem on even ranks only
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int num_ptcls = 0;
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", picparts->nelems());
  PS::kkGidView element_gids("element_gids", picparts->nelems());
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  if (rank % 2 == 0) {
    const int ppe = 100;
    Omega_h::parallel_for(picparts->nelems(), OMEGA_H_LAMBDA(const int& i) {
      ptcls_per_elem(i) = ppe;
      element_gids(i) = mesh_element_gids[i];
    });
    num_ptcls = ppe * picparts->nelems();
  }
  else {
    Omega_h::parallel_for(picparts->nelems(), OMEGA_H_LAMBDA(const int& i) {
      ptcls_per_elem(i) = 0;
      element_gids(i) = mesh_element_gids[i];
    });

  }

  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  const int C = 32;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, C);

  PS* ptcls = new pumipic::SellCSigma<Particle>(policy, sigma, V, picparts->nelems(),
                                                num_ptcls, ptcls_per_elem, element_gids);

  printImb(ptcls);


  //Balance particles
  balancePtcls(picparts, ptcls, balancer);
  printImb(ptcls);
  balancePtcls(picparts, ptcls, balancer);

  double imb = printImb(ptcls);
  if (imb > 1.5)
    ++fail;
  return fail;
}

void balancePtcls(pumipic::Mesh& picparts, PS* ptcls, pumipic::ParticleBalancer& balancer) {
  int comm_rank = picparts.comm()->rank();
  const int ps_capacity = ptcls->capacity();
  PS::kkLidView new_elems("ps_elem_ids", ps_capacity);
  PS::kkLidView new_parts("ps_process_ids", ps_capacity);
  auto is_safe = picparts.safeTag();
  auto owners = picparts.entOwners(picparts.dim());
  auto vals = ptcls->get<0>();
  auto setValues = PS_LAMBDA(const int elm, const int ptcl, const bool mask) {
    if (mask) {
      vals(ptcl) = comm_rank;
      const auto safe = is_safe[elm];
      new_elems(ptcl) = elm;
      if (!safe)
        new_parts(ptcl) = owners[elm];
      else
        new_parts(ptcl) = comm_rank;
    }
    else {
      vals(ptcl) = -1;
      new_elems(ptcl) = -1;
      new_parts(ptcl) = comm_rank;
    }
  };
  pumipic::parallel_for(ptcls, setValues);
  balancer.repartition(picparts, ptcls, 1.05, new_elems, new_parts);

  ptcls->migrate(new_elems, new_parts);
}

double printImb(PS* ptcls) {
  int np = ptcls->nPtcls();
  int min_p, max_p, tot_p;
  MPI_Reduce(&np, &min_p, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&np, &max_p, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&np, &tot_p, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if (comm_rank == 0) {
    float avg = tot_p / comm_size;
    float imb = max_p / avg;
    printf("Ptcl LB <max, min, avg, imb>: %d %d %.3f %.3f\n", max_p, min_p, avg, imb);
    return imb;
  }
  //All non 0 ranks return 1.0 (perfect imbalance)
  return 1.0;
}
