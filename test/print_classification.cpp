#include <fstream>

#include <Omega_h_file.hpp>  //gmsh
#include <Omega_h_for.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_array.hpp>
#include <pumipic_library.hpp>

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 3) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition file prefix>\n", 
              argv[0]);
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  if (comm_size > 1) {
    fprintf(stderr, "[ERROR] Only supports running in serial\n");
    return EXIT_FAILURE;
  }
  //Load mesh
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.world());
  int dim = mesh.dim();
  int ne = mesh.nelems();
  
  //Check off all class ids with elements and count number of elements with each class id
  auto class_ids = mesh.get_array<Omega_h::ClassId>(dim, "class_id");
  //Note: Assuming max(classid) < number of elements
  Omega_h::Write<Omega_h::LO> class_checks(ne,0);
  Omega_h::Write<Omega_h::LO> class_size(ne,0);
  auto checkClassIds = OMEGA_H_LAMBDA(const Omega_h::LO& id) {
    const Omega_h::ClassId c_id = class_ids[id];
    class_checks[c_id] = 1;
    Kokkos::atomic_fetch_add(&(class_size[c_id]), 1);
  };
  Omega_h::parallel_for(ne,checkClassIds,"checkClassIds");

  //Reduction to count the number of class Ids
  Omega_h::LO nclasses = Omega_h::get_sum(Omega_h::LOs(class_checks));
  //Exclusive sum to set part ids per class id (ignoring 0s in array)
  Omega_h::LOs class_ptn = Omega_h::offset_scan(Omega_h::LOs(class_checks));
  
  //Get max part
  Omega_h::LO max_ne = get_max(Omega_h::LOs(class_size));
  //Set 0s to number of elements to compute minimum
  auto setEmptyToMax = OMEGA_H_LAMBDA(const Omega_h::LO& i) {
    if (class_size[i] ==0)
      class_size[i] = ne;
  };
  Omega_h::parallel_for(ne, setEmptyToMax, "setEmptyToMax");
  Omega_h::LO min_ne = get_max(Omega_h::LOs(class_size));

  
  //Print partition stats
  Omega_h::Real avg = ne*1.0/nclasses;
  printf("Partition Stats: parts: %d max: %d min: %d avg: %.3f imb: %.3f\n",
         nclasses, max_ne, min_ne, avg, max_ne/avg);

  //Write partition to file
  Omega_h::HostRead<Omega_h::LO> host_class_owners(class_ptn);
  char filename[200];
  sprintf(filename, "%s.cpn",argv[2]);
  std::ofstream in_str(filename);
  if (!in_str) {
    fprintf(stderr, "Cannot open file %s\n", filename);
    return EXIT_FAILURE;
  }
  in_str<<nclasses<<'\n';
  for (int i = 0; i < ne-1; ++i) {
    if (host_class_owners[i] != host_class_owners[i+1])
      in_str << i << ' ' << host_class_owners[i] << '\n';
  }

  return EXIT_SUCCESS;
}
