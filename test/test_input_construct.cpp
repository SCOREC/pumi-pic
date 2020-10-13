#include <fstream>

#include <Omega_h_file.hpp>  //gmsh
#include <pumipic_mesh.hpp>
#include <Omega_h_for.hpp>

bool constructFullBFS(Omega_h::Mesh&, Omega_h::Write<Omega_h::LO> owners);
bool constructMinNone(Omega_h::Mesh&, Omega_h::Write<Omega_h::LO> owners);
bool constructBFSFull(Omega_h::Mesh&, Omega_h::Write<Omega_h::LO> owners);
bool constructBFSBFS(Omega_h::Mesh&, Omega_h::Write<Omega_h::LO> owners);
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
    printf("Full mesh loaded with <v e f r> %d %d %d %d\n", mesh.nverts(), mesh.nedges(),
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

  int fail = 0;
  if (!constructFullBFS(mesh, owner)) {
    fprintf(stderr, "constructFullBFS failed on rank %d\n",rank);
    ++fail;
  }
  if (!constructMinNone(mesh, owner)) {
    fprintf(stderr, "constructMinNone failed on rank %d\n",rank);
    ++fail;
  }
  // if (!constructBFSFull(mesh, owner)) {
  //   fprintf(stderr, "constructBFSFull failed on rank %d\n",rank);
  //   ++fail;
  // }
  // if (!constructBFSBFS(mesh, owner)) {
  //   fprintf(stderr, "constructBFSBFS failed on rank %d\n",rank);
  //   ++fail;
  // }

  if (argc >= 4 && !constructClassMinBFS(mesh, argv[3])) {
    fprintf(stderr, "constructClassMinBFS failed on rank %d\n",rank);
    ++fail;
  }

  if (fail == 0 && rank == 0)
    printf("All tests passed\n");
  return fail;
}

void BFS(int nents, Omega_h::Adj bridge2elems,Omega_h::LOs visited,
         Omega_h::Write<Omega_h::LO> visited_next) {
  auto meshBFS = OMEGA_H_LAMBDA(const Omega_h::LO bridge_id) {
    const auto deg = bridge2elems.a2ab[bridge_id + 1] - bridge2elems.a2ab[bridge_id];
    const auto firstElm = bridge2elems.a2ab[bridge_id];
    bool is_visited_here = false;
    for (int j = 0; j < deg; ++j) {
      const auto elm = bridge2elems.ab2b[firstElm+j];
      if (visited[elm])
        is_visited_here = true;
    }
    const int loops = deg*is_visited_here;
    for (int j = 0; j < loops; ++j) {
      const auto elm = bridge2elems.ab2b[firstElm+j];
      visited_next[elm] = true;
    }
  };
  Omega_h::parallel_for(nents, meshBFS,"meshBFS");
}

bool constructFullBFS(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner) {
  // Create Picparts with full buffer, BFS safe zone (3 layers)
  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::FULL,
                       pumipic::Input::BFS);
  input.bridge_dim = mesh.dim()-1;
  input.safeBFSLayers = 3;
  pumipic::Mesh picparts(input);

  bool ret = true;
  //Check entity counts of full buffer
  for (int i = 0; i <= mesh.dim(); ++i) {
    if (picparts.nents(i) != mesh.nents(i)) {
      ret = false;
      fprintf(stderr, "constructFullBFS failed because mesh entity count does not match"
              "full mesh on dimension %d. (%d != %d)\n", i, picparts.nents(i), mesh.nents(i));
    }
  }

  //Check safe zone BFS layers from core
  int rank = picparts.comm()->rank();
  Omega_h::Write<Omega_h::LO> is_visited(picparts.nelems(), 0);
  Omega_h::Write<Omega_h::LO> is_visited_next(picparts.nelems(), 0);
  const auto initVisit = OMEGA_H_LAMBDA( Omega_h::LO elem_id) {
    const Omega_h::LO own = owner[elem_id] == rank;
    is_visited[elem_id] = is_visited_next[elem_id] = own;
  };
  Omega_h::parallel_for(picparts.nelems(), initVisit, "initVisit");

  const auto bridge2elems = mesh.ask_up(input.bridge_dim, mesh.dim());
  for (int i = 0; i < input.safeBFSLayers; ++i) {
    BFS(mesh.nents(input.bridge_dim), bridge2elems, Omega_h::LOs(is_visited), is_visited_next);
    auto copyVisit = OMEGA_H_LAMBDA(Omega_h::LO elm_id) {
      is_visited[elm_id] = is_visited_next[elm_id];
    };
    Omega_h::parallel_for(picparts.nelems(), copyVisit, "copyVisit");
  }
  Omega_h::Write<Omega_h::LO> fails(1,0);
  Omega_h::LOs safeTag = picparts.safeTag();
  auto compareSafeAndVisit = OMEGA_H_LAMBDA(const int elem) {
    if (is_visited[elem] != safeTag[elem])
      fails[0] = 1;
  };
  Omega_h::parallel_for(picparts.nelems(), compareSafeAndVisit, "compareSafeAndVisit");
  return ret;
}
bool constructMinNone(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner) {
  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::MINIMUM,
                       pumipic::Input::NONE);

  pumipic::Mesh picparts(input);
  int rank = picparts.comm()->rank();
  Omega_h::Write<Omega_h::LO> fail(2, 0);
  Omega_h::LOs safe_tag = picparts.safeTag();
  auto checkSafeIsZero = OMEGA_H_LAMBDA(const Omega_h::LO& id) {
    if (safe_tag[id])
      fail[0] = 1;
    // if (owner[id] != rank)
    //   fail[1] = 1;
  };
  Omega_h::parallel_for(picparts.nelems(), checkSafeIsZero);
  Omega_h::HostWrite<Omega_h::LO> host_fail(fail);
  bool ret = true;
  if (host_fail[0] == 1) {
    ret = false;
    fprintf(stderr, "constructMinNone failed because safe zone is set\n");
  }
  if (host_fail[1] == 1) {
    ret = false;
    fprintf(stderr, "constructMinNone failed because buffer contains non core elements\n");
  }
  return ret;
}

bool constructBFSFull(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner) {
  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::BFS,
                       pumipic::Input::FULL);
  input.bufferBFSLayers = 3;
  input.safeBFSLayers = 2;

  pumipic::Mesh picparts(input);

  std::stringstream ss;
  ss << "constructBFSFull_r" << picparts.comm()->rank();
  std::string s = ss.str();
  Omega_h::vtk::write_parallel(s, picparts.mesh(), picparts.dim());


  bool ret = true;
  return ret;
}

bool constructBFSBFS(Omega_h::Mesh& mesh, Omega_h::Write<Omega_h::LO> owner) {
  pumipic::Input input(mesh, pumipic::Input::PARTITION, owner, pumipic::Input::BFS,
                       pumipic::Input::BFS);
  pumipic::Mesh picparts(input);

  bool ret = true;
  return ret;
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
