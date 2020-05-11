#include <fstream>

#include <Omega_h_file.hpp>  //gmsh

#include <pumipic_library.hpp>

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 3) {
    if (!rank)
      fprintf(stderr, "Usage: mpirun -np <num_parts> %s <mesh> <partition file prefix>\n",
              argv[0]);
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  if (comm_size == 1) {
      fprintf(stderr, "This tool must be run in parallel with the number of ranks equal to the"
              "target partition.\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.world());
  int dim = mesh.dim();
  Omega_h::Read<Omega_h::GO> global_ids = mesh.globals(dim);
  Omega_h::HostRead<Omega_h::GO> global_ids_h(global_ids);
  int nge = mesh.nglobal_ents(dim);
  int* owners = new int[nge];
  for (int i = 0; i < nge; ++i)
    owners[i] = 0;
  for (int i = 0; i < mesh.nelems(); ++i)
    owners[global_ids_h[i]] = rank;
  int* recv_owners = new int[nge];
  MPI_Reduce(owners, recv_owners, nge, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    char filename[200];
    sprintf(filename, "%s_%d.ptn",argv[2],comm_size);
    std::ofstream in_str(filename);
    if (!in_str) {
      fprintf(stderr, "Cannot open file %s\n", filename);
      return EXIT_FAILURE;
    }
    for (int i = 0; i < nge; ++i) {
      in_str << recv_owners[i] << '\n';
    }
  }
  Omega_h::vtk::write_parallel("partition", &mesh, dim);
  delete [] recv_owners;
  delete [] owners;

  return EXIT_SUCCESS;
}
