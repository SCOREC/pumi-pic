
#include "pumipic_adjacency.hpp"
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "GitrmMesh.hpp"
#include "unit_tests.hpp"

#include <Omega_h_library.hpp>
#include "pumipic_kktypes.hpp"
#include "pumipic_adjacency.hpp"
#include <psTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>
#include <MemberTypes.h>
#include <Distribute.h>
#include <Kokkos_Core.hpp>
#include "pumipic_library.hpp"


using particle_structs::fp_t;
using particle_structs::lid_t;
using particle_structs::Vector3d;
using particle_structs::SellCSigma;
using particle_structs::MemberTypes;

namespace o = Omega_h;
namespace p = pumipic;

//TODO remove mesh argument, once Singleton gm is used
GitrmParticles::GitrmParticles(o::Mesh &m, const int np):
  mesh(m) {
  //defineParticles(np);
}

GitrmParticles::~GitrmParticles(){
  delete scs;
}

// Initialized in only one element
void GitrmParticles::defineParticles(const int elId, const int numPtcls) {

  o::Int ne = mesh.nelems();

  SCS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  //Element gids is left empty since there is no partitioning of the mesh yet
  SCS::kkGidView element_gids;
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    ptcls_per_elem(i) = 0;
    if (i == elId)
      ptcls_per_elem(i) = numPtcls;
  });
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
    if (np > 0)
      printf("ppe[%d] %d\n", i, np);
  });

  //'sigma', 'V', and the 'policy' control the layout of the SCS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  //Create the particle structure
  scs = new SellCSigma<Particle>(policy, sigma, V, ne, numPtcls,
                   ptcls_per_elem, element_gids);

}

void GitrmParticles::initImpurityPtcls(const o::LO numPtcls, o::Real theta, o::Real phi, 
    const o::Real r, const o::LO maxLoops, const o::Real outer) {
  o::Write<o::LO> elemAndFace(3, -1); 
  o::LO initEl = -1;
  findInitBdryElemId(theta, phi, r, initEl, elemAndFace, maxLoops, outer);
  defineParticles(initEl, numPtcls);
  setPtclIds(scs);

  // TODO set initial prev_pos, vel, etc
  
  setImpurityPtclInitCoords(elemAndFace);
}

// spherical coordinates (wikipedia), radius r=1.5m, inclination theta[0,pi] from the z dir,
// azimuth angle phi[0, 2π) from the Cartesian x-axis (so that the y-axis has phi = +90°).
void GitrmParticles::findInitBdryElemId(o::Real theta, o::Real phi, const o::Real r,
     o::LO &initEl, o::Write<o::LO> &elemAndFace, const o::LO maxLoops, 
     const o::Real outer){

  o::LO debug = 4;

  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);

  const auto down_r2fs = down_r2f.ab2b;
  const auto dual_faces = dual.ab2b;
  const auto dual_elems = dual.a2ab;

  theta = theta * o::PI / 180.0;
  phi = phi * o::PI / 180.0;
  
  const o::Real x = r * sin(theta) * cos(phi);
  const o::Real y = r * sin(theta) * sin(phi);
  const o::Real z = r * cos(theta);

  o::Real endR = r + outer; //meter, to be outside of the domain
  const o::Real xe = endR * sin(theta) * cos(phi);
  const o::Real ye = endR * sin(theta) * sin(phi);
  const o::Real ze = endR * cos(theta);

  printf("\nDirection:x,y,z: %f %f %f\n xe,ye,ze: %f %f %f\n", x,y,z, xe,ye,ze);

  // Beginning element id of this x,y,z
  auto lamb = OMEGA_H_LAMBDA(const int elem) {
    auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
    auto M = p::gatherVectors4x3(coords, tetv2v);

    const o::Vector<3> orig{x,y,z};
    o::Vector<4> bcc;
    p::find_barycentric_tet(M, orig, bcc);
    if(p::all_positive(bcc, 0)) {
      elemAndFace[0] = elem;
      if(debug > 3)
        printf("ORIGIN detected in elem %d \n", elem);
    }
  };
  o::parallel_for(mesh.nelems(), lamb, "init_impurity_ptcl1");
  o::HostRead<o::LO> elemId_bh(elemAndFace);
  printf("ELEM_beg %d \n", elemId_bh[0]);

  OMEGA_H_CHECK(elemId_bh[0] >= 0);

  // Search final elemAndFace on bdry, on 1 thread on device(issue [] on host) 
  o::Write<o::Real> xpt(1, -1); 
  auto lamb2 = OMEGA_H_LAMBDA(const int e) {
    auto elem = elemAndFace[0];
    const o::Vector<3> dest{xe, ye, ze};
    o::Vector<3> orig{x,y,z};   
    o::Vector<4> bcc;
    bool found = false;
    o::LO loops = 0;

    while (!found) {

      if(debug > 4)
        printf("\n****ELEM %d : ", elem);

      // Destination should be outisde domain
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);

      p::find_barycentric_tet(M, dest, bcc);
      if(p::all_positive(bcc, 0)) {
        Omega_h_fail("Wrong guess of destination in initImpurityPtcls");
      }

      // Start search
      auto dface_ind = dual_elems[elem];
      const auto beg_face = elem *4;
      const auto end_face = beg_face +4;
      o::LO fIndex = 0;

      for(auto iface = beg_face; iface < end_face; ++iface) {
        const auto face_id = down_r2fs[iface];

        o::Vector<3> xpoint = o::zero_vector<3>();
        const auto face = p::get_face_of_tet(mesh2verts, coords, elem, fIndex);
        bool detected = p::line_triangle_intx_simple(face, orig, dest, xpoint);
        if(debug > 4) {
          printf("iface %d faceid %d detected %d\n", iface, face_id, detected);             
        }

        if(detected && side_is_exposed[face_id]) {
          found = true;
          elemAndFace[1] = elem;
          elemAndFace[2] = face_id;

          for(o::LO i=0; i<3; ++i)
            xpt[i] = xpoint[i];

          if(debug) {
            printf("faceid %d detected on exposed\n",  face_id);
          }
          break;
        } else if(detected && !side_is_exposed[face_id]) {
          auto adj_elem  = dual_faces[dface_ind];
          elem = adj_elem;
          if(debug >4) {
            printf("faceid %d detected on interior; next elm %d\n", face_id, elem);
          }
          break;
        }
        if(!side_is_exposed[face_id]){
          ++dface_ind;
        }
        ++fIndex;
      } // faces

      if(loops > maxLoops) {
          Omega_h_fail("Tried maxLoops iterations in initImpurityPtcls");
          break;
      }
      ++loops;
    }
  };
  o::parallel_for(1, lamb2, "init_impurity_ptcl2");

  o::HostRead<o::Real> xpt_h(xpt);

  o::HostRead<o::LO> elemId_fh(elemAndFace);
  initEl = elemId_fh[1];
  printf("ELEM_final %d xpt: %.3f %.3f %.3f\n", elemId_fh[1], xpt_h[0], xpt_h[1], xpt_h[2]);
  OMEGA_H_CHECK(elemId_fh[0] >= 0);

}

void GitrmParticles::setImpurityPtclInitCoords(o::Write<o::LO> &elemAndFace) {

  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);

  const auto down_r2fs = down_r2f.ab2b;
  const auto dual_faces = dual.ab2b;
  const auto dual_elems = dual.a2ab;

  //Set particle coordinates. Initialized only on one face. TODO confirm this ? 
  auto x_scs_d = scs->get<PTCL_POS>();
  auto x_scs_prev_d = scs->get<PTCL_POS_PREV>();
  auto vel_d = scs->get<PTCL_VEL>();

  srand(time(NULL));

  auto init = SCS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(elem == elemAndFace[1]) {
      const auto faceId = elemAndFace[2];
      const auto fv2v = o::gather_verts<3>(face_verts, faceId);
      const auto face = p::gatherVectors3x3(coords, fv2v);
      o::Vector<3> fnorm;
      p::find_face_normal(faceId, coords, face_verts, fnorm);
      //Inverse dir
      o::Vector<3> finv = -1*fnorm;
      o::Vector<3> pos;

      p::find_face_centroid(faceId, coords, face_verts, pos);
      // Move inwards a bit, then scatter
      auto rnd = (double)(std::rand())/RAND_MAX - 0.5;
      o::LO ind = (rnd < 0.3)?0:1;
      ind = (rnd > 0.6)?2:ind;

      o::Vector<3> scatter = pos + rnd * (pos -face[ind]);
      auto rnd2 = (double)(std::rand())/RAND_MAX;

      o::Vector<3> ppos = pos + 0.0001* finv + rnd2*scatter;
      pos = ppos + 0.0001* finv;
      if(elem%500 == 0)
        printf("elm %d xyz %f %f %f\n", elem, pos[0], pos[1], pos[2]);

      for(int i=0; i<3; i++) {
        x_scs_prev_d(pid,i) = ppos[i];
        //TODO ??
        x_scs_d(pid,i) = pos[i] ;
        vel_d(pid,i) = std::rand();
      }
    }

  };
  scs->parallel_for(init);
}
