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

#include "mpi.h"

int main(int argc, char** argv) {
  Omega_h::Library lib = Omega_h::Library(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 3) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition file prefix>\n", 
              argv[0]);
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  Omega_h::Mesh mesh = Omega_h::gmsh::read(argv[1], lib.world());
  int dim = mesh.dim();
  Omega_h::Read<Omega_h::GO> global_ids = mesh.globals(dim);
  int nge = mesh.nglobal_ents(dim);
  int* owners = new int[nge];
  for (int i = 0; i < nge; ++i)
    owners[i] = 0;
  for (int i = 0; i < mesh.nelems(); ++i)
    owners[global_ids[i]] = rank;
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
