#include <Omega_h_mesh.hpp>
#include "Omega_h_build.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_for.hpp"
#include "pumipic_kktypes.hpp"
#include "pumipic_adjacency.hpp"
#include "pumipic_library.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include "pumipic_mesh.hpp"
#include <fstream>
#include <cstdlib>
#include <ctime>
#define NUM_ITERATIONS 30

using particle_structs::fp_t;
using particle_structs::lid_t;
using particle_structs::Vector3d;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;
using particle_structs::distribute_name;
using particle_structs::elemCoords;

namespace o = Omega_h;
namespace p = pumipic;
typedef MemberTypes<Vector3d, Vector3d, int> Particle;
typedef SellCSigma<Particle> SCS;


void setPtclIds(SCS* scs) {
  auto pid_d = scs->get<2>();
  auto setIDs = SCS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    pid_d(pid) = pid;
  };
  scs->parallel_for(setIDs);
}

void updatePtclPositions(SCS* scs) {
  auto x_scs_d = scs->get<0>();
  auto xtgt_scs_d = scs->get<1>();
  auto updatePtclPos = SCS_LAMBDA(const int&, const int& pid, const bool&) {
    x_scs_d(pid,0) = xtgt_scs_d(pid,0);
    x_scs_d(pid,1) = xtgt_scs_d(pid,1);
    x_scs_d(pid,2) = xtgt_scs_d(pid,2);
    xtgt_scs_d(pid,0) = 0;
    xtgt_scs_d(pid,1) = 0;
    xtgt_scs_d(pid,2) = 0;
  };
  scs->parallel_for(updatePtclPos);
}

void rebuild(SCS* scs, o::LOs elem_ids) {
  updatePtclPositions(scs);
  const int scs_capacity = scs->capacity();
  auto pid_d =  scs->get<2>();
  auto printElmIds = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0 ) {//&& elem_ids[pid] >= 0) 
      //printf(">> Particle remains %d \n", pid);
      printf("rebuild:elem_ids[%d] %d ptcl %d\n", pid, elem_ids[pid], pid_d(pid));
    }
  };
  //scs->parallel_for(printElmIds);
  SCS::kkLidView scs_elem_ids("scs_elem_ids", scs_capacity);
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void)e;
    scs_elem_ids(pid) = elem_ids[pid];
  };
  scs->parallel_for(lamb);
  scs->rebuild(scs_elem_ids);
}

void search(o::Mesh& mesh, SCS* scs, int iter=0) {
  assert(scs->nElems() == mesh.nelems());
  Omega_h::LO maxLoops = 100;
  const auto scsCapacity = scs->capacity();
  o::Write<o::LO> elem_ids(scsCapacity,-1);
  o::Write<o::Real>xpoints(scsCapacity*3, 0, "xpoints");
  o::Write<o::LO>xface_ids(scsCapacity, -1, "xpoints");
  auto x_scs = scs->get<0>();
  auto xtgt_scs = scs->get<1>();
  auto pid_scs = scs->get<2>();
  bool isFound = p::search_mesh<Particle>(mesh, scs, x_scs, xtgt_scs, pid_scs, 
    elem_ids, xpoints, xface_ids, maxLoops);
  assert(isFound);
  //gp.collisionPoints = o::Real(xpoints);
  //gp.collisionPointFaceIds = o::LOs(xface_ids);
  rebuild(scs, elem_ids);
}

void setPtclInitialCoords(o::Mesh& mesh, SCS* scs, o::Real fac=1.0e-6,
 bool debug=false) {

  const auto down_r2f = mesh.ask_down(3, 2);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto down_r2fs = down_r2f.ab2b;
  //const auto side_is_exposed = mark_exposed_sides(&mesh);
  //const auto dual = mesh.ask_dual();
  //const auto dual_faces = dual.ab2b;
  //const auto dual_elems = dual.a2ab;
  auto x_scs_prev_d = scs->get<0>();
  int scsCapacity = scs->capacity();
  o::HostWrite<o::Real> rnd1(scsCapacity);
  o::HostWrite<o::Real> rnd2(scsCapacity);
  std::srand(time(NULL));
  for(auto i=0; i<scsCapacity; ++i) {
    rnd1[i] = (double)(std::rand())/RAND_MAX;
    rnd2[i] = (double)(std::rand())/RAND_MAX;
  }
  o::Write<o::Real>rand1(rnd1);
  o::Write<o::Real>rand2 (rnd2);
  printf("done rand numbers\n");

  auto lambda = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      //auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      //auto M = gatherVectors4x3(coords, tetv2v);
      //auto dface_ind = dual_elems[elem];
      o::LO findex = 0;
      for(auto iface = elem*4; iface < (elem+1)*4; ++iface) {
        const auto faceId = down_r2fs[iface];
        //o::LO exposed = side_is_exposed[faceId];
        const auto fv2v = o::gather_verts<3>(face_verts, faceId);
        const auto face = p::gatherVectors3x3(coords, fv2v);
        //auto fcent = p::find_face_centroid(faceId, coords, face_verts);
        auto tcent = p::centroid_of_tet(elem, mesh2verts, coords); 
        auto rn1 = rand1[pid];
        auto rn2 = rand2[pid];
        //on face
        auto pos1 = face[0]+ rn1*(face[1] - face[0]);
        auto pos = face[2] + rn2*(pos1 - face[2]);

        //auto fnorm = p::find_face_normal(faceId, elem, coords, mesh2verts, 
        //                                face_verts, down_r2fs);
        auto dvec = tcent-pos;
        auto dirCent = o::normalize(dvec);
        o::Real delta = fac * rn1;
        pos = pos + delta*dirCent;

        for(int i=0; i<3; i++) {
          x_scs_prev_d(pid,i) = pos[i];
        }

        if(debug)
          printf("elm %d:pos %.4f %.4f %.4f \n", elem, pos[0], pos[1], pos[2]);
        //if(!exposed)
        //  ++dface_ind;
        ++findex;
      } //faces
    } //mask
  };
  scs->parallel_for(lambda);
}

// not in good shape
void push(SCS* scs) {


  auto pos_d = scs->get<0>();
  auto new_pos_d = scs->get<1>();
  const auto capacity = scs->capacity();
  
  fp_t distance = 0.5;
  fp_t dx = 0.1;
  fp_t dy = 0.2;
  fp_t dz = 0.3;

  fp_t disp[4] = {distance,dx,dy,dz};
  p::kkFpView disp_d("direction_d", 4);
  p::hostToDeviceFp(disp_d, disp);

  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    fp_t delta[3];
    delta[0] = disp_d(0)*disp_d(1);
    delta[1] = disp_d(0)*disp_d(2);
    delta[2] = disp_d(0)*disp_d(3);
    new_pos_d(pid,0) = pos_d(pid,0) + delta[0];
    new_pos_d(pid,1) = pos_d(pid,1) + delta[1];
    new_pos_d(pid,2) = pos_d(pid,2) + delta[2];
  };
  scs->parallel_for(lamb);

}

int main(int argc, char** argv) {
  Kokkos::initialize(argc,argv);
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  if(argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " <mesh>\n";
    exit(1);
  }

  auto mesh = o::gmsh::read(argv[1], world);
  //Omega_h::Mesh mesh(&lib);

  //Omega_h::build_from_elems2verts(&mesh, OMEGA_H_SIMPLEX, 3, 
  //  Omega_h::LOs({0, 1, 2, 3}), 4);

  int elem = 0; //id
  int nPtcls = 10; 
  if(argc >2) 
    nPtcls = atof(argv[2]);
  int ne = mesh.nelems();
  
  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  SCS::kkGidView element_gids("elem_gids", ne);
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    if (i==elem) {
      ptcls_per_elem[i] = nPtcls;
      printf("ppe[%d] %d\n", i, nPtcls);
    }
  });

  const int sigma = INT_MAX;
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  SellCSigma<Particle>* scs = new SellCSigma<Particle>(policy, sigma, V, ne, 
                      nPtcls, ptcls_per_elem, element_gids);

  printf("Setting initial positions \n");
  o::Real fac=1.0e-6;
  setPtclInitialCoords(mesh, scs, fac);
  setPtclIds(scs);

  for(int iter=1; iter<=NUM_ITERATIONS; iter++) {
    if(scs->nPtcls() == 0) {
      printf("No particles remain... exiting\n");
      break;
    }
    printf("-----iter----- %d\n", iter);
    push(scs);
    search(mesh, scs, iter);
  }
  //cleanup
  delete scs;
  Kokkos::Cuda::finalize();
  return 0;
}
