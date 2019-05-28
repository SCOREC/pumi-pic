#include <fstream>

#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 5) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename> "
              "<# of safe layers> <# of buffer layers>\n", argv[0]);
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int safe_layers = atoi(argv[3]);
  int ghost_layers = atoi(argv[4]);

  //**********Load the mesh in serial everywhere*************//
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.self());
  int dim = mesh.dim();
  int ne = mesh.nents(dim);
  if (rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh.nverts(), mesh.nedges(), 
           mesh.nfaces(), mesh.nelems());

  //********* Load the partition vector ***********//
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
  
  pumipic::Mesh picparts(mesh,owner,ghost_layers, safe_layers);

  char vtk_name[100];
  sprintf(vtk_name, "picpart%d", rank);
  Omega_h::vtk::write_parallel(vtk_name, picparts.mesh(), dim);
  
  return 0;
}
