#include <Omega_h_mesh.hpp>
#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pseudoXGCmTypes.hpp"
#include "gyroScatter.hpp"
#include <fstream>
#include "pumipic_version.hpp"

typedef Kokkos::DefaultExecutionSpace ExeSpace;

void setPtclIds(PS* ptcls) {
  auto pid_d = ptcls->get<2>();
  auto setIDs = PS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    pid_d(pid) = pid;
  };
  ps::parallel_for(ptcls, setIDs);
}

void setInitialPtclCoords(p::Mesh& picparts, PS* ptcls) {
  //get centroid of parent element and set the child particle coordinates
  //most of this is copied from Omega_h_overlay.cpp get_cell_center_location
  //It isn't clear why the template parameter for gather_[verts|vectors] was
  //sized eight... maybe something to do with the 'Overlay'.  Given that there
  //are four vertices bounding a tet, I'm setting that parameter to four below.
  o::Mesh* mesh = picparts.mesh();
  auto cells2nodes = mesh->get_adj(o::FACE, o::VERT).ab2b;
  auto nodes2coords = mesh->coords();
  //set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = o::gather_verts<3>(cells2nodes, o::LO(e));
      auto vtxCoords = o::gather_vectors<3,2>(nodes2coords, elmVerts);
      auto center = average(vtxCoords);
        for(int i=0; i<2; i++)
          x_ps_d(pid,i) = center[i];
    }
  };
  ps::parallel_for(ptcls, lamb);
}

o::Mesh readMesh(char* meshFile, o::Library& lib) {
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self());
  } else {
    std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

o::LOs modifyMappings(o::Mesh* mesh, o::LOs& inmap) {
  const auto gnr = gyro_num_rings;
  const auto gppr = gyro_points_per_ring;
  o::Write<o::LO> map(inmap.size());
  o::parallel_for(mesh->nverts(), OMEGA_H_LAMBDA(const o::LO& v) {
    const auto vtxIdx = v*gnr*gppr;
    for(int ring=0; ring<gnr; ring++) {
       const auto ringIdx = ring*gppr;
       for(int pt=0; pt<gppr; pt++) {
         const auto ptIdx = 3*(vtxIdx+ringIdx+pt);
         for(int elmVtx=0; elmVtx<3; elmVtx++) {
           const auto mappedIdx = ptIdx + elmVtx;
           if(v != 3) { //vtx 3 is at the center of the plate
             map[mappedIdx] = 2; // vtx 2 is in the top left corner
           } else {
             map[mappedIdx] = inmap[mappedIdx];
           }
         }
      }
    }
  });
  return o::LOs(map);
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if( argc != 2 ) {
    std::cout << "Usage: " << argv[0] << " <mesh> " << "\n";
    exit(1);
  }
  printf("VERSION: %s\n", pumipic::pumipic_version());
  auto full_mesh = readMesh(argv[1], lib);
  Omega_h::HostWrite<Omega_h::LO> host_owners(full_mesh.nelems());
  for (int i = 0; i < full_mesh.nelems(); ++i)
    host_owners[i] = 0;
  Omega_h::Write<Omega_h::LO> owner(host_owners);
  pumipic::Input input(full_mesh, pumipic::Input::PARTITION, owner, pumipic::Input::FULL,
                       pumipic::Input::BFS);
  pumipic::Mesh picparts(input);
  for (int i = 0; i <= full_mesh.dim(); ++i)
    assert(picparts.nents(i) == full_mesh.nents(i));

  //Create Picparts with the full mesh
  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts(); //caching adjacency info
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  printf("mesh_element_gids.size() %d picparts.dim() %d\n",
      mesh_element_gids.size(), picparts.dim());

  if (comm_rank == 0)
    printf("Mesh loaded with <v e f r> %d %d %d %d\n", mesh->nverts(), mesh->nedges(),
           mesh->nfaces(), mesh->nelems());

  //Build gyro avg mappings
  const auto radius = .2;
  const auto rings = 2;
  const auto pointsPerRing = 6;
  const auto theta = 15;
  setGyroConfig(radius, rings, pointsPerRing, theta);
  if (!comm_rank) printGyroConfig();
  Omega_h::LOs forward_map;
  Omega_h::LOs backward_map;
  createGyroRingMappings(mesh, forward_map, backward_map);

  //modify the mappings to only scatter values from the central vertex
  auto fwd_map_centerOnly = modifyMappings(mesh,forward_map);
  auto bkwd_map_centerOnly = modifyMappings(mesh,backward_map);

  /* Particle data */
  const auto ne = mesh->nelems();
  const auto numPtcls = 1;
  if (comm_rank == 0)
    fprintf(stderr, "number of elements %d number of particles %d\n",
            ne, numPtcls);
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  //place one particle in element 0
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = mesh_element_gids[i];
    ptcls_per_elem(i) = (i == 0);
    printInfo("ppe[%d] %d\n", i, ptcls_per_elem(i));
  });

  const int sigma = INT_MAX; // full sorting
  const int V = 32;
  Kokkos::TeamPolicy<ExeSpace> policy = pumipic::TeamPolicyAuto(10000, 32);
  //Create the particle structure
  PS* ptcls = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
                                       ptcls_per_elem, element_gids);
  setInitialPtclCoords(picparts, ptcls);
  setPtclIds(ptcls);

  //define parameters controlling particle motion
  auto fwdTagName = "ptclToMeshScatterFwd";
  auto bkwdTagName = "ptclToMeshScatterBkwd";
  mesh->add_tag(o::VERT, fwdTagName, 1, o::Reals(mesh->nverts(), 0));
  mesh->add_tag(o::VERT, bkwdTagName, 1, o::Reals(mesh->nverts(), 0));

  gyroScatter(mesh,ptcls,fwd_map_centerOnly, fwdTagName);
  gyroScatter(mesh,ptcls,bkwd_map_centerOnly, bkwdTagName);

  auto fwdTagVals = mesh->get_array<o::Real>(o::VERT, fwdTagName);
  Omega_h::parallel_for(mesh->nverts(), OMEGA_H_LAMBDA(const int& i) {
    if(i==3) {
      OMEGA_H_CHECK(o::are_close(fwdTagVals[i],2.0));
    } else if(i==2) {
      OMEGA_H_CHECK(o::are_close(fwdTagVals[i],12.0));
    } else if(i==8) {
      OMEGA_H_CHECK(o::are_close(fwdTagVals[i],0.0));
    } else {
      OMEGA_H_CHECK(o::are_close(fwdTagVals[i],2.0/3.0));
    }
  });
  auto bkwdTagVals = mesh->get_array<o::Real>(o::VERT, bkwdTagName);
  Omega_h::parallel_for(mesh->nverts(), OMEGA_H_LAMBDA(const int& i) {
      OMEGA_H_CHECK(o::are_close(fwdTagVals[i],bkwdTagVals[i]));
  });

  //cleanup
  delete ptcls;

  Omega_h::vtk::write_parallel("pseudoPush_tf", mesh, picparts.dim());
  if (!comm_rank)
    fprintf(stderr, "done\n");
  return 0;
}
