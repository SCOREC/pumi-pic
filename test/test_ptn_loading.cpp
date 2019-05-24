#include <fstream>
#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_for.hpp"
#include "Omega_h_file.hpp"  //gmsh
#include "Omega_h_tag.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_array.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_mark.hpp"
#include "Omega_h_class.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_reduce.hpp"


#include "unit_tests.hpp"
#include "pumipic_utils.hpp"
#include "pumipic_part_construct.hpp"
#include "mpi.h"

#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  //Kokkos::initialize(argc, argv);
  Omega_h::Library lib = Omega_h::Library(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 5) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename> "
              "<# of safe layers> <# of buffer layers>\n", argv[0]);
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int safe_layers = atoi(argv[3]);
  int ghost_layers = atoi(argv[4]);

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

  Omega_h::Mesh picpart(&lib);
  pumipic::constructPICParts(mesh, owner, safe_layers, ghost_layers, &picpart);

  char vtk_name[80];
  sprintf(vtk_name, "picpart%d",rank);
  Omega_h::vtk::write_parallel(vtk_name, &picpart, dim);
  //Kokkos::finalize();
  return 0;
}
