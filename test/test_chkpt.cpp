#include <fstream>

#include <particle_structs.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>
#include <pumipic_lb.hpp>


typedef pumipic::MemberTypes<int> Particle;
typedef pumipic::ParticleStructure<Particle> PS;

PS* createPS(pumipic::Mesh& picparts);

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
  PS* ptcls = createPS(picparts);

  //write checkpoint
  ptcls->checkpoint("/path/to/output/file");

  return 0;
}

PS* createPS(pumipic::Mesh& picparts) {
  //Create 100 particles/elem
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int num_ptcls = 0;
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", picparts->nelems());
  PS::kkGidView element_gids("element_gids", picparts->nelems());
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  const int ppe = 100;
  Omega_h::parallel_for(picparts->nelems(), OMEGA_H_LAMBDA(const int& i) {
    ptcls_per_elem(i) = ppe;
    element_gids(i) = mesh_element_gids[i];
  });
  num_ptcls = ppe * picparts->nelems();

  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(picparts->nelems(), 32);

  PS* ptcls = new ps::CabM<Particle>(policy, picparts->nelems(),
                                                num_ptcls, ptcls_per_elem, element_gids);
  return ptcls;
}
