#include <fstream>

#include <Omega_h_file.hpp>  //gmsh
#include <Omega_h_for.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_array.hpp>
#include <pumipic_library.hpp>

int getNextClass(int start, Omega_h::HostWrite<Omega_h::LO> owners);

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 5 && argc != 6) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition file prefix> <num_ranks (-1 -> num classids)>"
              " <minimum number of elements per rank> [load balancing tol]\n", argv[0]);
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
  printf("Mesh Loaded <v, e, f");
  if (dim == 3)
    printf(", r");
  printf("> %d %d %d", mesh.nverts(), mesh.nedges(), mesh.nfaces());
  if (dim == 3)
    printf(" %d", mesh.nents(3));
printf("\n");
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
  printf("Mesh has %d classifications with max classification %d.\n"
         "Building partition with %d ranks\n", nclasses, max_class, comm_size);

  //Exclusive sum to set part ids per class id (ignoring 0s in array)
  Omega_h::LOs class_ptn = Omega_h::offset_scan(Omega_h::LOs(class_checks));
  Omega_h::HostRead<Omega_h::LO> class_ptn_h(class_ptn);

  Omega_h::HostRead<Omega_h::LO> class_size_h(class_size);

  Omega_h::HostWrite<Omega_h::LO> ptn_sizes(comm_size);
  Omega_h::HostWrite<Omega_h::LO> ptn_starts(comm_size);
  Omega_h::HostWrite<Omega_h::LO> ptn_ends(comm_size);

  for(int i=0; i<comm_size; i++)
    ptn_sizes[i] = 0;

  Omega_h::HostWrite<Omega_h::LO> host_class_owners(max_class+1);

  if (comm_size > nclasses) {
    fprintf(stderr, "[WARNING] Too may ranks requested, reducing from %d to %d\n",
            comm_size, nclasses);
    comm_size = nclasses;
  }

  Omega_h::LO classes_per_rank = (nclasses)/(comm_size);
  Omega_h::LO extra_classes = (nclasses)%(comm_size);
#ifdef PP_DEBUG
  printf("Calculation of class per rank: %d + %d\n", classes_per_rank, extra_classes);
#endif

  int tot_class = 0;
  int cur_class = 0, num_class;
  int cur_size;
  //For each rank
  for (int cur_rank = 0; cur_rank < comm_size; ++cur_rank) {
    num_class = 0;
    cur_size = 0;
    ptn_starts[cur_rank] = cur_class;
    //While over class ids
    while (cur_class <= max_class &&
           (num_class < classes_per_rank + (extra_classes > 0) || cur_size < min_elems)) {
      host_class_owners[cur_class] = -1;
      if (class_size_h[cur_class] > 0) {
        num_class++;
        cur_size += class_size_h[cur_class];
        host_class_owners[cur_class] = cur_rank;
      }
      ++cur_class;
    }
    ptn_ends[cur_rank] = cur_class - 1;
    ptn_sizes[cur_rank] = cur_size;
    tot_class += num_class;
    int over = num_class - (classes_per_rank + (extra_classes > 0));
    if (over > 0) { //If we assigned too many classes recalculate class per rank
      int rem_class = nclasses - tot_class;
      int rem_comm = comm_size - cur_rank - 1;
      if (rem_comm > rem_class) {
        int diff_comm = rem_comm - rem_class;
        comm_size -= diff_comm;
        rem_comm = rem_class;
        fprintf(stderr, "[WARNING] Too many ranks requested, reducing from %d to %d\n",
                comm_size + diff_comm, comm_size);
      }
      if (rem_comm > 0) {
        classes_per_rank = rem_class / (rem_comm);
      extra_classes = rem_class % (rem_comm);
      }
       else
        classes_per_rank = extra_classes = 0;

#ifdef PP_DEBUG
      printf("Recalculation of class per rank: %d + %d\n", classes_per_rank, extra_classes);
#endif

    }
    else if (extra_classes > 0) //If we just assigned a remainder class, decrease by one
      --extra_classes;
  }
#ifdef PP_DEBUG
  printf("%d of %d classes allocated\n", tot_class, nclasses);
#endif
  //Get max part
  Omega_h::LO max_ne = get_max(Omega_h::LOs(Omega_h::Write<Omega_h::LO>(ptn_sizes)));
  Omega_h::LO min_ne = get_min(Omega_h::LOs(Omega_h::Write<Omega_h::LO>(ptn_sizes)));

  //Print partition stats
  Omega_h::Real avg = ne*1.0/comm_size;
  printf("Partition Stats: parts: %d max: %d min: %d avg: %.3f imb: %.3f\n",
         comm_size, max_ne, min_ne, avg, max_ne/avg);

  //Load Balancing
  if (argc == 6) {
    Omega_h::Real tol = atof(argv[5]);
    int iters = 10;
    for (int i = 0; i < iters; ++i) {
      //Forward LB
      int cur_class = getNextClass(0, host_class_owners);
      for (int cur_rank = 0; cur_rank < comm_size - 1; cur_rank++) {
        cur_class = ptn_ends[cur_rank] + 1;
        int next_class = getNextClass(cur_class + 1, host_class_owners);
        while (cur_class < ptn_ends[cur_rank + 1] && ptn_sizes[cur_rank] < avg * (1-tol)) {
          if (host_class_owners[cur_class] != -1) {
            int owner = host_class_owners[cur_class];
            int size = class_size_h[cur_class];
            ptn_sizes[owner] -= size;
            ptn_sizes[cur_rank] += size;
            host_class_owners[cur_class] = cur_rank;
            ptn_ends[cur_rank] = cur_class;
            ptn_starts[owner] = cur_class + 1;
          }
          cur_class = next_class;
          next_class = getNextClass(cur_class + 1, host_class_owners);
        }
      }
    }
    //Get max part
    max_ne = get_max(Omega_h::LOs(Omega_h::Write<Omega_h::LO>(ptn_sizes)));
    min_ne = get_min(Omega_h::LOs(Omega_h::Write<Omega_h::LO>(ptn_sizes)));

    //Print partition stats
    printf("Partition Stats Post LB: parts: %d max: %d min: %d avg: %.3f imb: %.3f\n",
           comm_size, max_ne, min_ne, avg, max_ne/avg);

#ifdef PP_DEBUG
    for (int i = 0; i < comm_size; ++i) {
      if (ptn_sizes[i] == max_ne) {
        printf("Process %d has max elements\n", i);
      }
      if (ptn_sizes[i] == min_ne) {
        printf("Process %d has min elements\n", i);
      }
    }
#endif
  }
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

int getNextClass(int start, Omega_h::HostWrite<Omega_h::LO> owners) {
  int c = start;
  while (c < owners.size() && owners[c] == -1)
    c++;
  return c;
}
