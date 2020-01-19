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
  if (argc != 5) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition file prefix> <num_ranks (-1 -> num classids)>"
              "<minimum number of elements per rank>\n", 
              argv[0]);
    return EXIT_FAILURE;
  }
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  if (comm_size > 1) {
    fprintf(stderr, "[ERROR] Only supports running in serial\n");
    return EXIT_FAILURE;
  }
  comm_size = atoi(argv[3]);
  int min_elems = atoi(argv[4]);
  //Load mesh
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.world());
  int dim = mesh.dim();
  int ne = mesh.nelems();
  
  //Check off all class ids with elements and count number of elements with each class id
  auto class_ids = mesh.get_array<Omega_h::ClassId>(dim, "class_id");

  Omega_h::ClassId max_class= Omega_h::get_max(class_ids);
  
  Omega_h::Write<Omega_h::LO> class_checks(max_class+2, 0);
  Omega_h::Write<Omega_h::LO> class_size(max_class+1, 0);
  auto checkClassIds = OMEGA_H_LAMBDA(const Omega_h::LO& id) {
    const Omega_h::ClassId c_id = class_ids[id];
    class_checks[c_id] = 1;
    Kokkos::atomic_fetch_add(&(class_size[c_id]), 1);
  };
  Omega_h::parallel_for(ne,checkClassIds,"checkClassIds");

  //Reduction to count the number of class Ids
  Omega_h::LO nclasses = Omega_h::get_sum(Omega_h::LOs(class_checks));
  if (comm_size == -1)
    comm_size = nclasses;

  //Exclusive sum to set part ids per class id (ignoring 0s in array)
  Omega_h::LOs class_ptn = Omega_h::offset_scan(Omega_h::LOs(class_checks));
  Omega_h::HostRead<Omega_h::LO> class_ptn_h(class_ptn);

  Omega_h::HostRead<Omega_h::LO> class_size_h(class_size);

  Omega_h::HostWrite<Omega_h::LO> ptn_sizes(comm_size);
  for(int i=0; i<comm_size; i++)
    ptn_sizes[i] = 0;

  Omega_h::HostWrite<Omega_h::LO> host_class_owners(max_class+1);

  int id = 0;
  int nc = 0;
  //filling is true if a current part is being filled with classifications
  bool filling = false;
  int outrank = 0;
  while (filling || (class_size_h[id] < min_elems)) {
    filling = true;
    host_class_owners[id] = -1;
    const Omega_h::LO my_class = class_ptn_h[id];
    const Omega_h::LO next_class = class_ptn_h[id+1];
    //Skip empty classifications
    if (my_class != next_class) {
      nc++;
      host_class_owners[id] = outrank;
      ptn_sizes[outrank] += class_size_h[id];
      if (ptn_sizes[outrank] >= min_elems) {
        outrank++;
        filling = false;
      }
    }
    ++id;
  }
  Omega_h::LO classes_per_rank = (nclasses-nc)/(comm_size-outrank);
  Omega_h::LO extra_classes = (nclasses-nc)%(comm_size-outrank);
  if (comm_size-outrank > nclasses-id) {
    fprintf(stderr,"[WARNING] Too many ranks for number of parts, reducing comm_size to fit classifications\n");
    comm_size = nclasses - id;
    extra_classes = 0;
    classes_per_rank = 1;
  }
  int start = outrank;
  int somecount = 0;
  for(; id<max_class+1; id++) {
    host_class_owners[id] = -1;
    const Omega_h::LO my_class = class_ptn_h[id];
    const Omega_h::LO next_class = class_ptn_h[id+1];
    //Skip empty classifications
    if (my_class != next_class) {
      somecount++;
      host_class_owners[id] = outrank;
      ptn_sizes[outrank] += class_size_h[id];
      if(somecount >= classes_per_rank + (outrank-start<extra_classes) ) {
        outrank++;
        somecount = 0;
      }
    }
  }
  
  //Get max part
  Omega_h::LO max_ne = get_max(Omega_h::LOs(Omega_h::Write<Omega_h::LO>(ptn_sizes)));
  Omega_h::LO min_ne = get_min(Omega_h::LOs(Omega_h::Write<Omega_h::LO>(ptn_sizes)));
  
  //Print partition stats
  Omega_h::Real avg = ne*1.0/comm_size;
  printf("Partition Stats: parts: %d max: %d min: %d avg: %.3f imb: %.3f\n",
         comm_size, max_ne, min_ne, avg, max_ne/avg);

  //Write partition to file
  char filename[200];
  sprintf(filename, "%s_%d.cpn",argv[2], comm_size);
  std::ofstream in_str(filename);
  if (!in_str) {
    fprintf(stderr, "Cannot open file %s\n", filename);
    return EXIT_FAILURE;
  }
  in_str<<max_class<<'\n';
  for (int i = 0; i < max_class+1; ++i) {
    if (host_class_owners[i] != -1)
      in_str << i << ' ' << host_class_owners[i] << '\n';
  }

  return EXIT_SUCCESS;
}
