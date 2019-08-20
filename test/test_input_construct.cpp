#include <fstream>

#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>

bool constructFullBFS(Omega_h::Mesh&, char* partition_file);
bool constructMinNone(Omega_h::Mesh&, char* partition_file);
bool constructClassMinBFS(Omega_h::Mesh&, char* class_file);

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (argc != 3 && argc != 4) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename> [classification partition]\n", argv[0]);
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
  int fail = 0;
  if (!constructFullBFS(mesh, argv[2])) {
    fprintf(stderr, "constructFullBFS failed on rank %d\n",rank);
    ++fail;
  }
  if (!constructMinNone(mesh, argv[2])) {
    fprintf(stderr, "constructMinNone failed on rank %d\n",rank);
    ++fail;
  }
  if (argc >= 4 && !constructClassMinBFS(mesh, argv[3])) {
    fprintf(stderr, "constructClassMinBFS failed on rank %d\n",rank);
    ++fail;
  }


  return fail;
}

bool constructFullBFS(Omega_h::Mesh& mesh, char* partition_file) {
  int ne = mesh.nelems();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  //********* Load the partition vector ***********//
  Omega_h::HostWrite<Omega_h::LO> host_owners(ne);
  std::ifstream in_str(partition_file);
  
  if (!in_str) {
    if (!rank)
      fprintf(stderr,"Cannot open file %s\n", partition_file);
    return EXIT_FAILURE;
  }
  int own;
  int index = 0;
  while(in_str >> own) 
    host_owners[index++] = own;
  //Owner of each element
  Omega_h::Write<Omega_h::LO> owner(host_owners);

  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::FULL,
                       pumipic::Input::BFS);
  input.bridge_dim = mesh.dim()-1;
  input.safeBFSLayers = 3;
  
  pumipic::Mesh picparts(input);

  for (int i = 0; i <= mesh.dim(); ++i) {
    if (picparts.nents(i) != mesh.nents(i))
      return false;
  }
  return true;
}
bool constructMinNone(Omega_h::Mesh& mesh, char* partition_file) {
  int ne = mesh.nelems();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //********* Load the partition vector ***********//
  Omega_h::HostWrite<Omega_h::LO> host_owners(ne);
  std::ifstream in_str(partition_file);
  if (!in_str) {
    if (!rank)
      fprintf(stderr,"Cannot open file %s\n", partition_file);
    return EXIT_FAILURE;
  }
  int own;
  int index = 0;
  while(in_str >> own) 
    host_owners[index++] = own;
  //Owner of each element
  Omega_h::Write<Omega_h::LO> owner(host_owners);

  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::MINIMUM,
                       pumipic::Input::NONE);
  
  pumipic::Mesh picparts(input);

  Omega_h::Write<Omega_h::LO> fail(1, 0);
  Omega_h::LOs safe_tag = picparts.safeTag();
  auto checkSafeIsZero = OMEGA_H_LAMBDA(const Omega_h::LO& id) {
    if (safe_tag[id])
      fail[0] = 1;
  };
  Omega_h::parallel_for(picparts.nelems(), checkSafeIsZero);
  Omega_h::HostWrite<Omega_h::LO> host_fail(fail);
  if (host_fail[0] == 1)
    return false;
  return true;
}

bool constructClassMinBFS(Omega_h::Mesh& mesh, char* class_file) {
  int ne = mesh.nelems();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  //********* Load the partition vector ***********//
  std::ifstream in_str(class_file);
  if (!in_str) {
    if (!rank)
      fprintf(stderr,"Cannot open file %s\n", class_file);
    return EXIT_FAILURE;
  }
  int size;
  in_str>>size;
  Omega_h::HostWrite<Omega_h::LO> host_owners(size+1);
  int cid, own;
  while(in_str >> cid >> own) 
    host_owners[cid] = own;
  //Owner of each element
  Omega_h::Write<Omega_h::LO> owner(host_owners);

  pumipic::Input input(mesh, pumipic::Input::CLASSIFICATION, owner, pumipic::Input::MINIMUM,
                       pumipic::Input::BFS);
  
  pumipic::Mesh picparts(input);

  Omega_h::Write<Omega_h::LO> fail(1);
  Omega_h::LOs safe_tag = picparts.safeTag();
  auto checkSafeIsZero = OMEGA_H_LAMBDA(const Omega_h::LO& id) {
    if (!safe_tag[id])
      fail[0] = 1;
  };
  Omega_h::parallel_for(picparts.nelems(), checkSafeIsZero);
  Omega_h::HostWrite<Omega_h::LO> host_fail(fail);
  if (host_fail[0] == 1)
    return false;
  return true;

}
