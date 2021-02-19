#include <fstream>

#include <particle_structs.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>
#include <pumipic_lb.hpp>


typedef pumipic::MemberTypes<int> Particle;
typedef pumipic::ParticleStructure<pumipic::SellCSigma<Particle>> PS;

void printImb(PS* ptcls);
void balancePtcls(pumipic::Mesh& picparts, PS* ptcls, pumipic::ParticleBalancer& balancer);


int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 4) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename> <num safe layers>\n", argv[0]);
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

  int fail = 0;

  Omega_h::HostWrite<Omega_h::LO> host_owners(ne);
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
  //Owner of each element
  Omega_h::Write<Omega_h::LO> owner(host_owners);
  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::BFS,
                       pumipic::Input::BFS);
  input.safeBFSLayers = atoi(argv[3]);
  pumipic::Mesh picparts(input);

  //Build Particle Balancer
  pumipic::ParticleBalancer balancer(picparts);

  //Create 100 particles/elem on even ranks only
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

  PS* ptcls = new PS(pumipic::SellCSigma<Particle>(policy, sigma, V,
                                                   picparts->nelems(),
                                                   num_ptcls, ptcls_per_elem,
                                                   element_gids));

  printImb(ptcls);


  //Balance particles
  balancePtcls(picparts, ptcls, balancer);

  balancePtcls(picparts, ptcls, balancer);

  auto globalIds = picparts.globalIds(picparts->dim());
  picparts->add_tag<Omega_h::GO>(picparts->dim(), "global_ids", 1, globalIds);
  char render_name[128];
  sprintf(render_name, "lb_%d", picparts.comm()->rank());
  Omega_h::vtk::write_parallel(render_name, picparts.mesh(), picparts.dim());


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
  printImb(ptcls);
}

void printImb(PS* ptcls) {
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
  }
}
