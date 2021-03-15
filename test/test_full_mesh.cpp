#include <fstream>

#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <pumipic_mesh.hpp>

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 3) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename>\n", argv[0]);
    MPI_Finalize();
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

  //********* Load the partition vector ***********//
  pumipic::Input input(mesh, argv[2], pumipic::Input::FULL, pumipic::Input::FULL);

  pumipic::Mesh picparts(input);

  for (int i = 0; i <= mesh.dim(); ++i) {
    if (mesh.nents(i) != picparts.mesh()->nents(i)) {
      fprintf(stderr, "Entity counts do not match on process %d for dimension %d (%d != %d)\n",
              rank, i, mesh.nents(i), picparts.mesh()->nents(i));
      return EXIT_FAILURE;
    }
  }

  //Test the global tag to be converted to global_serial
  Omega_h::TagBase const* new_tag = picparts->get_tagbase(3, "global_serial");

  Omega_h::GOs global_old = mesh.get_array<Omega_h::GO>(3, "global");
  Omega_h::GOs global_new = picparts->get_array<Omega_h::GO>(3, "global_serial");
  Omega_h::Write<Omega_h::LO> fails(1,0);
  auto checkGlobals = OMEGA_H_LAMBDA(const Omega_h::LO ent) {
    if (global_old[ent] != global_new[ent]) {
      printf("global ids do not match on entity %d [%ld != %ld]", ent, global_old[ent],
        global_old[ent]);
      fails[0] = 1;
    }
  };
  Omega_h::parallel_for(mesh.nelems(), checkGlobals, "checkGlobals");

  char vtk_name[100];
  sprintf(vtk_name, "picpart%d", rank);
  Omega_h::vtk::write_parallel(vtk_name, picparts.mesh(), dim);

  return Omega_h::HostWrite<Omega_h::LO>(fails)[0];
}
