#include <fstream>
#include <iostream>
#include <cmath>
#include <utility>

#include <Omega_h_for.hpp>
#include <Omega_h_file.hpp>  //gmsh
#include <Omega_h_array.hpp>
#include <Omega_h_mesh.hpp>

#include <pumipic_mesh.hpp>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  Omega_h::Library lib = Omega_h::Library(&argc, &argv);
  
  int rank = lib.world()->rank();;
  if (argc != 3) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename>\n", argv[0]);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  //**********Load the mesh in serial everywhere*************//
  Omega_h::Mesh mesh = Omega_h::gmsh::read(argv[1], lib.self());
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
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  int own;
  int index = 0;
  while(in_str >> own) 
    host_owners[index++] = own;
  //Owner of each element
  Omega_h::Write<Omega_h::LO> owner(host_owners);

  //********* Construct the PIC parts *********//
  pumipic::Mesh picparts(mesh, owner, 3, 1);

  //Create an array with initial values of 1
  Omega_h::Write<Omega_h::LO> comm_array = picparts.createCommArray(dim, 1, 1);

  picparts.reduceCommArray(dim, pumipic::Mesh::SUM_OP, comm_array);

  //Check that each entry is equal to the number of ranks
  Omega_h::HostWrite<Omega_h::LO> host_array(comm_array);
  bool success = true;
  for (int i = picparts.offset_ents_per_rank_per_dim[dim][rank];
       i < picparts.offset_ents_per_rank_per_dim[dim][rank]; ++i)
    if (host_array[i] != picparts.num_buffers())
      success = false;

  if (!success) {
    fprintf(stderr, "Sum operation failed on %d\n", lib.world()->rank());
    return EXIT_FAILURE;
  }
  return 0;
}
