#include <fstream>

#include <Omega_h_for.hpp>
#include <Omega_h_file.hpp>
#include <pumipic_mesh.hpp>
#include <Kokkos_Core.hpp>


bool minOwnership(pumipic::Mesh& picparts, int dim);
bool sumEntities(pumipic::Mesh& picparts, int dim);
bool fullBufferTest(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner, int dim);

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int rank = lib.world()->rank();;
  if (argc != 3) {
    if (!rank)
      fprintf(stderr, "Usage: %s <mesh> <partition filename>\n", argv[0]);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

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
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  int own;
  int index = 0;
  while(in_str >> own) 
    host_owners[index++] = own;
  //Owner of each element
  Omega_h::Write<Omega_h::LO> owner(host_owners);

  for (int i = 0; i <= mesh.dim(); ++i) {
    if (!fullBufferTest(mesh, owner, i))
      printf("fullBufferTest on dimension %d failed on rank %d\n", i, rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  
  //********* Construct the PIC parts *********//
  pumipic::Mesh picparts(mesh, owner, 1, 0);

  for (int i = 0; i <= picparts.dim(); ++i) {
    if (!minOwnership(picparts, i))
      printf("minOwnership on dimension %d failed on rank %d\n", i, rank);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i <= 0/*picparts.dim()*/; ++i) {
    if (!sumEntities(picparts, i))
      printf("sumEntities on dimension %d failed on rank %d\n", i, rank);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  Omega_h::Write<Omega_h::Real> max_comm = picparts.createCommArray(0, 1, 0.0);
  auto setLIDVtx = OMEGA_H_LAMBDA(Omega_h::LO vtx_id) {
    max_comm[vtx_id] = vtx_id;
  };
  Omega_h::parallel_for(picparts.nents(0), setLIDVtx);

  MPI_Barrier(MPI_COMM_WORLD);
  Omega_h::Write<Omega_h::LO> fail(1, 0);
  auto checkVtxMax = OMEGA_H_LAMBDA(Omega_h::LO vtx_id) {
    if (max_comm[vtx_id] < vtx_id)
      fail[0] = 1;
  };
  Omega_h::parallel_for(picparts.mesh()->nents(0), checkVtxMax);
  
  Omega_h::HostWrite<Omega_h::LO> fail_host(fail);
  if (fail_host[0]) {
    fprintf(stderr, "Max reduce failed on %d\n", rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  //Create an array with initial values of 0
  Omega_h::Write<Omega_h::LO> comm_array = picparts.createCommArray(dim, 3, 0);

  Omega_h::Read<Omega_h::LO> owners = picparts.entOwners(3);
  auto setMyElmsToOne = OMEGA_H_LAMBDA(Omega_h::LO elm_id) {
    for (int i = 0; i < 3; ++i)
      comm_array[elm_id*3 + i] = (owners[elm_id] == rank);
  };
  Omega_h::parallel_for(picparts.mesh()->nelems(), setMyElmsToOne);

  picparts.reduceCommArray(dim, pumipic::Mesh::SUM_OP, comm_array);

  //Check that each entry is equal to 1
  Omega_h::HostWrite<Omega_h::LO> host_array(comm_array);
  Omega_h::HostRead<Omega_h::LO> elm_offsets(picparts.nentsOffsets(3));
  bool success = true;
  for (int i = 0; i < host_array.size(); ++i) {
    if (host_array[i] != 1)
      success = false;
  }
  
  if (!success) {
    fprintf(stderr, "Multielement comm operation failed on %d\n", lib.world()->rank());
  }

  return 0;
}

bool minOwnership(pumipic::Mesh& picparts, int dim) {
  int rank = picparts.comm()->rank();
  Omega_h::Write<Omega_h::LO> owner_comm = picparts.createCommArray(dim, 1, INT_MAX);

  Omega_h::LOs vtx_owners = picparts.entOwners(dim);
  auto setOwnedVtx = OMEGA_H_LAMBDA(Omega_h::LO vtx_id) {
    if (vtx_owners[vtx_id] == rank)
      owner_comm[vtx_id] = rank;
  };
  Omega_h::parallel_for(picparts.mesh()->nents(dim), setOwnedVtx);

  picparts.reduceCommArray(dim, pumipic::Mesh::MIN_OP, owner_comm);

  Omega_h::Write<Omega_h::LO> fail(1, 0);
  auto checkVtx = OMEGA_H_LAMBDA(Omega_h::LO vtx_id) {
    if (owner_comm[vtx_id] != vtx_owners[vtx_id])
      fail[0] = 1;
  };
  Omega_h::parallel_for(picparts.mesh()->nents(0), checkVtx);
  
  Omega_h::HostWrite<Omega_h::LO> fail_host(fail);
  if (fail_host[0]) {
    return false;
  }
  return true;
}

bool sumEntities(pumipic::Mesh& picparts, int dim) {
  //Sum the occurences of each entity
  int rank = picparts.comm()->rank();
  Omega_h::Write<Omega_h::LO> sum_comm = picparts.createCommArray(dim, 1, 1);

  picparts.reduceCommArray(dim, pumipic::Mesh::SUM_OP, sum_comm);

  Omega_h::Write<Omega_h::Real> contribution_comm = picparts.createCommArray(dim, 3, 0.0);

  auto setContribution = OMEGA_H_LAMBDA(Omega_h::LO id) {
    for (int i = 0; i < 3; i++)
      contribution_comm[id*3 + i] = 1.0 / sum_comm[id];
  };
  Omega_h::parallel_for(picparts.nents(dim), setContribution, "setContribution");
  
  picparts.reduceCommArray(dim, pumipic::Mesh::SUM_OP, contribution_comm);
  
  Omega_h::Write<Omega_h::LO> fail(1, 0);
  auto checkVtx = OMEGA_H_LAMBDA(Omega_h::LO vtx_id) {
    for (int i = 0; i < 3; ++i) 
      if (Kokkos::fabs(contribution_comm[vtx_id*3+i] - 1.0) > .00001)
        fail[0] = 1;
  };
  Omega_h::parallel_for(picparts.mesh()->nents(0), checkVtx);
  
  Omega_h::HostWrite<Omega_h::LO> fail_host(fail);
  if (fail_host[0]) {
    return false;
  }
  return true;
}

bool fullBufferTest(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner, int dim) {
  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::FULL,
                       pumipic::Input::FULL);

  pumipic::Mesh picparts(input);

  Omega_h::Write<Omega_h::LO> comm_arr = picparts.createCommArray(dim, 1, 1);

  picparts.reduceCommArray(dim, pumipic::Mesh::SUM_OP, comm_arr);

  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  Omega_h::Write<Omega_h::LO> fail(1, 0);
  auto checkCommArr = OMEGA_H_LAMBDA(const Omega_h::LO& id) {
    if (comm_arr[id] != comm_size)
      fail[0] = 1;
  };
  Omega_h::parallel_for(picparts.mesh()->nents(dim), checkCommArr);

  Omega_h::HostWrite<Omega_h::LO> fail_host(fail);
  if (fail_host[0]) {
    return false;
  }

  return true;
}
