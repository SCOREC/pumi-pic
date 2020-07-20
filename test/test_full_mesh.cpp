#include <fstream>

#include <Omega_h_file.hpp>

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

  char vtk_name[100];
  sprintf(vtk_name, "picpart%d", rank);
  Omega_h::vtk::write_parallel(vtk_name, picparts.mesh(), dim);

  return EXIT_SUCCESS;
}
