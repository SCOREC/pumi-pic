#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_bbox.hpp>
#include "pumipic_adjacency.hpp"
#include <random>
#include "team_policy.hpp"

#define ELEMENT_SEED 1024*1024
#define PARTICLE_SEED 512*512

namespace o = Omega_h;
namespace p = pumipic;

using p::fp_t;
using p::Vector3d;
/* Define particle types
   0 = current position
   1 = pushed position
   2 = ids
   3 = direction of motion
 */
typedef p::MemberTypes<Vector3d, Vector3d, int, Vector3d> Particle;
typedef p::ParticleStructure<Particle> PS;
typedef Kokkos::DefaultExecutionSpace ExeSpace;

int setSourceElements(o::Mesh mesh, PS::kkLidView ppe, const int numPtcls) {
  auto numPpe = numPtcls / mesh.nelems();
  auto numPpeR = numPtcls % mesh.nelems();
  auto cells2nodes = mesh.ask_down(mesh.dim(), o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  o::LO lastElm = mesh.nelems() - 1;
  o::parallel_for(mesh.nelems(), OMEGA_H_LAMBDA(const int i) {
      ppe[i] = numPpe + (i < numPpeR);
    });
  Omega_h::LO totPtcls = 0;
  Kokkos::parallel_reduce(ppe.size(), OMEGA_H_LAMBDA(const int i, Omega_h::LO& lsum) {
      lsum += ppe[i];
    }, totPtcls);
  assert(totPtcls == numPtcls);
  return totPtcls;
}


PS* create_particle_structure(o::Mesh mesh, p::lid_t numPtcls) {
  Omega_h::Int ne = mesh.nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    element_gids(i) = i;
  });
  int actualParticles = setSourceElements(mesh, ptcls_per_elem, numPtcls);
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const int& i) {
    const int np = ptcls_per_elem(i);
  });

  //'sigma', 'V', and the 'policy' control the layout of the PS structure
  //in memory and can be ignored until performance is being evaluated.  These
  //are reasonable initial settings for OpenMP.
  const int sigma = INT_MAX; // full sorting
  const int V = 1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy = pumipic::TeamPolicyAuto(10000, 32);
  //Create the particle structure
  return new p::SellCSigma<Particle>(policy, sigma, V, ne, actualParticles,
                                     ptcls_per_elem, element_gids);
}

void init2DInternal(o::Mesh mesh, PS* ptcls) {
  //Randomly distrubite particles within each element (uniformly within the element)
  //Create a deterministic generation of random numbers on the host with 2 number per particle

  o::HostWrite<o::Real> rand_num_per_ptcl(3*ptcls->capacity());
  std::default_random_engine generator(PARTICLE_SEED);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < ptcls->capacity(); ++i) {
    o::Real x = dist(generator);
    o::Real y = dist(generator);
    o::Real ang = dist(generator);
    if (x+y > 1) {
      x = 1-x;
      y = 1-y;
    }
    rand_num_per_ptcl[3*i] = x;
    rand_num_per_ptcl[3*i+1] = y;
    rand_num_per_ptcl[3*i+2] = ang * 2* M_PI;
  }
  o::Write<o::Real> rand_nums(rand_num_per_ptcl);
  auto cells2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  //set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>();
  auto x_ps_tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  auto motion = ptcls->get<3>();
  o::Reals elmArea = measure_elements_real(&mesh);
  
  auto lamb = PS_LAMBDA(const int e, const int pid, const int mask) {
    if(mask > 0) {
      auto elmVerts = o::gather_verts<3>(cells2nodes, o::LO(e));
      auto vtxCoords = o::gather_vectors<3,2>(nodes2coords, elmVerts);
      o::Real r1 = rand_nums[3*pid];
      o::Real r2 = rand_nums[3*pid+1];
      o::Real r3 = rand_nums[3*pid+2];
      // X = A + r1(B-A) + r2(C-A)
      for (int i = 0; i < 2; i++) {
        x_ps_d(pid,i) = vtxCoords[0][i] + r1 * (vtxCoords[1][i] - vtxCoords[0][i])
                                        + r2 * (vtxCoords[2][i] - vtxCoords[0][i]);
        x_ps_tgt(pid, i) = x_ps_d(pid, i);
      }
      x_ps_tgt(pid,2) = x_ps_d(pid,2) = 0;

      auto ptclOrigin = p::makeVector2(pid, x_ps_d);
      Omega_h::Vector<3> faceBcc;
      p::barycentric_tri(elmArea[e], vtxCoords, ptclOrigin, faceBcc);
      assert(p::all_positive(faceBcc, 1e-8));
      if (!p::all_positive(faceBcc, 1e-8)) printf("FAILURE\n");
      motion(pid, 0) = cos(r3);
      motion(pid, 1) = sin(r3);
      motion(pid, 2) = 0;
      pids(pid) = pid;
    }
  };
  ps::parallel_for(ptcls, lamb);
}

void init_2D_edges(o::Mesh mesh, PS* ptcls) {
  o::HostWrite<o::Real> rand_num_per_ptcl(2*ptcls->capacity());
  o::HostWrite<o::LO> rand_ints_per_ptcl(3*ptcls->capacity());
  std::default_random_engine generator(PARTICLE_SEED);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  srand(PARTICLE_SEED);
  for (int i = 0; i < ptcls->capacity(); ++i) {
    o::LO type = rand() % 3;
    rand_ints_per_ptcl[3*i] = type;
    if (type == 0) {
      //Generate particle on vertex and move along an edge
      //Which vertex index of triangle
      rand_ints_per_ptcl[3*i+1] = rand() % 3;
      //Which edge from that vertex (will be modded in parallel_for)
      rand_ints_per_ptcl[3*i+2] = rand();
    }
    else if (type == 1) {
      //Generate particle on edge and move along that edge
      //Which edge index of triangle
      rand_ints_per_ptcl[3*i+1] = rand() % 3;
      //Which direction along the edge
      rand_ints_per_ptcl[3*i+2] = 2*(rand() % 2)-1;
      //How far along the edge to spawn particle
      rand_num_per_ptcl[2*i] = dist(generator);
    }
    else {
      //Generate particle in triangle and move through a vertex
      //Location in triangle
      o::Real x = dist(generator);
      o::Real y = dist(generator);
      if (x+y > 1) {
        x = 1-x;
        y = 1-y;
      }
      rand_num_per_ptcl[2*i] = x;
      rand_num_per_ptcl[2*i+1] = y;
      //Which vertex to move towards
      rand_ints_per_ptcl[3*i+1] = rand() % 3;
    }
  }
  
  //Transfer random numbers to device
  o::Write<o::Real> rand_nums(rand_num_per_ptcl);
  o::Write<o::LO> rand_ints(rand_ints_per_ptcl);

  auto cells2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
  auto cells2edges = mesh.ask_down(o::FACE, o::EDGE).ab2b;
  auto coords = mesh.coords();
  auto verts2edges = mesh.ask_up(o::VERT, o::EDGE);
  auto edges2verts = mesh.ask_down(o::EDGE, o::VERT).ab2b;
  //set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>();
  auto x_ps_tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  auto motion = ptcls->get<3>();
  o::Reals elmArea = measure_elements_real(&mesh);
  
  auto lamb = PS_LAMBDA(const int e, const int pid, const int mask) {
    if(mask > 0) {
      const o::LO type = rand_ints[3*pid];
      if (type == 0) {
        //Set particle position on vertex
        const o::LO vert_index = rand_ints[3*pid+1];
        const o::LO vert_id = cells2nodes[3*e+vert_index];
        x_ps_d(pid, 0) = coords[2*vert_id];
        x_ps_d(pid, 1) = coords[2*vert_id+1];
        x_ps_d(pid, 2) = 0;

        //Pick edge to move along
        const o::LO firstEdge = verts2edges.a2ab[vert_id];
        const o::LO nedge = verts2edges.a2ab[vert_id+1] - firstEdge;
        const o::LO edgeIndex = rand_ints[3*pid+2] % nedge;
        const o::LO edgeID = verts2edges.ab2b[firstEdge + edgeIndex];
        const auto edgeVerts = o::gather_verts<2>(edges2verts, edgeID);
        const auto edgeCoords = o::gather_vectors<2,2>(coords, edgeVerts);
        const o::Vector<2> dir = o::normalize(edgeCoords[1] - edgeCoords[0]);
        motion(pid, 0) = dir[0];
        motion(pid, 1) = dir[1];
        motion(pid, 2) = 0;
      }
      else if (type == 1) {
        //Set particle on edge
        const o::LO edgeIndex = rand_ints[3*pid+1];
        const o::LO edgeID = cells2edges[3*e+edgeIndex];
        const auto edgeVerts = o::gather_verts<2>(edges2verts, edgeID);
        const auto edgeCoords = o::gather_vectors<2,2>(coords, edgeVerts);
        const o::Vector<2> dir = edgeCoords[1] - edgeCoords[0];
        const o::Real dist = rand_nums[2*pid];
        for (int i = 0; i < 2; ++i)
          x_ps_d(pid, i) = edgeCoords[0][i] + dist * dir[i];
        x_ps_d(pid, 2) = 0;

        //Choose direction along edge
        const o::LO flip = rand_ints[3*pid+2];
        const o::Vector<2> ndir = o::normalize(dir);
        for (int i = 0; i < 2; ++i)
          motion(pid, i) =  flip * ndir[i];
        motion(pid, 2) = 0;
      }
      else {
        //Place particle in triangle
        auto elmVerts = o::gather_verts<3>(cells2nodes, o::LO(e));
        auto vtxCoords = o::gather_vectors<3,2>(coords, elmVerts);
        o::Real r1 = rand_nums[2*pid];
        o::Real r2 = rand_nums[2*pid+1];
        // X = A + r1(B-A) + r2(C-A)
        for (int i = 0; i < 2; i++)
          x_ps_d(pid,i) = vtxCoords[0][i] + r1 * (vtxCoords[1][i] - vtxCoords[0][i])
            + r2 * (vtxCoords[2][i] - vtxCoords[0][i]);
        x_ps_d(pid,2) = 0;

        //Move particle towards a vertex
        const o::LO vertIndex = rand_ints[3*pid + 1];
        const o::LO vertID = cells2nodes[3*e+vertIndex];
        o::Vector<2> vertCoord;
        for (int i = 0; i < 2; ++i) vertCoord[i] = coords[2*vertID + i];
        const auto ptclPos = p::makeVector2(pid, x_ps_d);
        const o::Vector<2> dir = o::normalize(vertCoord - ptclPos);
        for (int i = 0; i < 2; ++i) motion(pid, i) = dir[i];
        motion(pid, 2) = 0;
        
      }
      //Ensure particle is created in the parent element
      const auto elmVerts = o::gather_verts<3>(cells2nodes, o::LO(e));
      const auto vtxCoords = o::gather_vectors<3,2>(coords, elmVerts);
      const auto ptclOrigin = p::makeVector2(pid, x_ps_d);
      Omega_h::Vector<3> faceBcc;
      p::barycentric_tri(elmArea[e], vtxCoords, ptclOrigin, faceBcc);
      assert(p::all_positive(faceBcc, 1e-8));      
    }
    pids(pid) = pid;
    for (int i = 0; i < 3; ++i)
      x_ps_tgt(pid, i) = x_ps_d(pid, i);
  };
  ps::parallel_for(ptcls, lamb);

}

void init_3D_edges(o::Mesh mesh, PS* ptcls) {
  const o::LO rpp = 4;
  const o::LO ipp = 4;
  o::HostWrite<o::Real> rand_num_per_ptcl(rpp*ptcls->capacity());
  o::HostWrite<o::LO> rand_ints_per_ptcl(ipp*ptcls->capacity());

  std::default_random_engine generator(PARTICLE_SEED);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  srand(PARTICLE_SEED);

  for (int i = 0; i < ptcls->capacity(); ++i) {
    o::LO spawnType = 0;//rand() % 4;
    o::LO tgtType = rand() % 2;
    rand_ints_per_ptcl[ipp*i] = spawnType;
    rand_ints_per_ptcl[ipp*i+1] = tgtType;
    if (spawnType == 0) {
      //Spawn on a vertex of the tet
      //The vertex index
      rand_ints_per_ptcl[ipp*i+2] = rand() % 4;
    }
    else if (spawnType == 1) {
      //Spawn on an edge of the tet
      //The edge index
      rand_ints_per_ptcl[ipp*i+2] = rand() % 6;
      //Portion along the edge
      rand_num_per_ptcl[rpp*i] = dist(generator);
    }
    else if (spawnType == 2) {
      //Spawn on a face of the tet
      //The face index
      rand_ints_per_ptcl[ipp*i+2] = rand() % 4;
      //The two parameters controlling the place in the triangle
      o::Real x = dist(generator);
      o::Real y = dist(generator);
      if (x+y > 1) {
        x = 1-x;
        y = 1-y;
      }
      rand_num_per_ptcl[rpp*i] = x;
      rand_num_per_ptcl[rpp*i+1] = y;
    }
    else if (spawnType == 3) {
      //Spawn in the interior of the tet
      //The three parameters that control location in the tet
      o::Real x = dist(generator);
      o::Real y = dist(generator);
      o::Real z = dist(generator);
      if (x+y > 1) {
        x = 1-x;
        y = 1-y;
      }
      if (y+z > 1) {
        o::Real tmp = z;
        z = 1 - x - y;
        y = 1 - tmp;
      }
      else if (x + y + z > 1) {
        o::Real tmp = z;
        z = x + y + z  - 1;
        x = 1 - y - tmp;
      }
      rand_num_per_ptcl[rpp*i] = x;
      rand_num_per_ptcl[rpp*i+1] = y;
      rand_num_per_ptcl[rpp*i+2] = z;
    }
    if (tgtType == 0) {
      //Move through a vertex
      rand_ints_per_ptcl[ipp*i+3] = rand() % 4;
    }
    else if (tgtType == 1) {
      //Move through an edge
      rand_ints_per_ptcl[ipp*i+3] = rand() % 6;
      rand_num_per_ptcl[rpp*i+3] = dist(generator);
    }
  }

  //Transfer random numbers to device
  o::Write<o::Real> rand_nums(rand_num_per_ptcl);
  o::Write<o::LO> rand_ints(rand_ints_per_ptcl);

  auto cells2verts = mesh.ask_down(3,0).ab2b;
  auto cells2edges = mesh.ask_down(3,1).ab2b;
  auto cells2faces = mesh.ask_down(3,2).ab2b;
  auto faces2verts = mesh.ask_down(2,0).ab2b;
  auto edges2verts = mesh.ask_down(1,0).ab2b;
  auto coords = mesh.coords();

  auto x_ps_d = ptcls->get<0>();
  auto x_ps_tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  auto motion = ptcls->get<3>();

  auto placeParticles = PS_LAMBDA(const int e, const int ptcl, const int mask) {
    if (mask > 0) {
      const o::LO spawnType = rand_ints[ipp*ptcl];
      if (spawnType == 0) {
        //Set particle position on vertex
        const o::LO vertIndex = rand_ints[ipp*ptcl+2];
        const o::LO vertID = cells2verts[4*e+vertIndex];
        x_ps_d(ptcl, 0) = coords[3*vertID];
        x_ps_d(ptcl, 1) = coords[3*vertID+1];
        x_ps_d(ptcl, 2) = coords[3*vertID+2];
      }
      else if (spawnType == 1) {
        //Set particle on edge
        const o::LO edgeIndex = rand_ints[ipp*ptcl+2];
        const o::LO edgeID = cells2edges[6*e+edgeIndex];
        const auto edgeVerts = o::gather_verts<2>(edges2verts, edgeID);
        const auto edgeCoords = o::gather_vectors<2,3>(coords, edgeVerts);
        const o::Vector<3> dir = edgeCoords[1] - edgeCoords[0];
        const o::Real dist = rand_nums[rpp*ptcl];
        for (int i = 0; i < 3; ++i)
          x_ps_d(ptcl, i) = edgeCoords[0][i] + dist * dir[i];
      }
      else if (spawnType == 2) {
        //Set particle on face
        const o::LO triIndex = rand_ints[ipp*ptcl+2];
        const o::LO triID = cells2faces[4*e + triIndex];
        auto faceVerts = o::gather_verts<3>(faces2verts, triID);
        auto vtxCoords = o::gather_vectors<3,3>(coords, faceVerts);
        o::Real r1 = rand_nums[rpp*ptcl];
        o::Real r2 = rand_nums[rpp*ptcl+1];
        // X = A + r1(B-A) + r2(C-A)
        for (int i = 0; i < 3; i++)
          x_ps_d(ptcl,i) = vtxCoords[0][i] + r1 * (vtxCoords[1][i] - vtxCoords[0][i])
            + r2 * (vtxCoords[2][i] - vtxCoords[0][i]);
      }
      else if (spawnType == 3) {
        //Set particle in the tet
        auto elmVerts = o::gather_verts<4>(cells2verts, o::LO(e));
        auto vtxCoords = o::gather_vectors<4,3>(coords, elmVerts);
        const o::Real s = rand_nums[rpp*ptcl];
        const o::Real t = rand_nums[rpp*ptcl + 1];
        const o::Real u = rand_nums[rpp*ptcl + 2];
        const o::Real a = 1 - s - t - u;
        for (int i = 0; i < 3; i++) {
          x_ps_d(ptcl, i) = a * vtxCoords[0][i] + s * vtxCoords[1][i]
            + t * vtxCoords[2][i] + u * vtxCoords[3][i];
        }
      }
      const o::LO tgtType = rand_ints[ipp*ptcl + 1];
      const o::Vector<3> ptclPos = p::makeVector3(ptcl, x_ps_d);
      o::Vector<3> tgtCoord;

      if (tgtType == 0) {
        const o::LO vertIndex = rand_ints[ipp*ptcl+3];
        const o::LO vertID = cells2verts[4*e+vertIndex];
        for (int i =0; i < 3; ++i) tgtCoord[i] = coords[3*vertID + i];
      }
      else if (tgtType == 1) {
        const o::LO edgeIndex = rand_ints[ipp*ptcl+3];
        const o::LO edgeID = cells2edges[6*e+edgeIndex];
        const auto edgeVerts = o::gather_verts<2>(edges2verts, edgeID);
        const auto edgeCoords = o::gather_vectors<2,3>(coords, edgeVerts);
        const o::Vector<3> dir = edgeCoords[1] - edgeCoords[0];
        const o::Real dist = rand_nums[rpp*ptcl+3];
        for (int i = 0; i < 3; ++i)
          tgtCoord[i] = edgeCoords[0][i] + dist * dir[i];
      }
      const o::Vector<3> dir = tgtCoord - ptclPos;
      if (norm(dir) != 0) {
        const o::Vector<3> ndir = normalize(dir);
        for (int i = 0; i < 3; ++i) motion(ptcl, i) = ndir[i];
      }
      else {
        for (int i = 0; i < 3; ++i) motion(ptcl, i) = 0;
      }
    }
    pids(ptcl) = ptcl;
    for (int i = 0; i < 3; ++i)
      x_ps_tgt(ptcl, i) = x_ps_d(ptcl, i);
  };
  p::parallel_for(ptcls, placeParticles, "placeParticles");
}

void init3DInternal(o::Mesh mesh, PS* ptcls) {
  //Randomly distrubite particles within each element (uniformly within the element)
  //Method taken from http://vcg.isti.cnr.it/jgt/tetra.htm
  //Create a deterministic generation of random numbers on the host with 2 number per particle

  o::HostWrite<o::Real> rand_num_per_ptcl(5*ptcls->capacity());
  std::default_random_engine generator(PARTICLE_SEED);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < ptcls->capacity(); ++i) {
    o::Real x = dist(generator);
    o::Real y = dist(generator);
    o::Real z = dist(generator);
    o::Real ang = dist(generator);
    o::Real r = dist(generator);
    if (x+y > 1) {
      x = 1-x;
      y = 1-y;
    }
    if (y+z > 1) {
      o::Real tmp = z;
      z = 1 - x - y;
      y = 1 - tmp;
    }
    else if (x + y + z > 1) {
      o::Real tmp = z;
      z = x + y + z  - 1;
      x = 1 - y - tmp;
    }
    rand_num_per_ptcl[5*i] = x;
    rand_num_per_ptcl[5*i+1] = y;
    rand_num_per_ptcl[5*i+2] = z;
    rand_num_per_ptcl[5*i+3] = ang * 2 * M_PI;
    rand_num_per_ptcl[5*i+4] = r * 2 - 1;
  }
  o::Write<o::Real> rand_nums(rand_num_per_ptcl);
  auto cells2nodes = mesh.ask_down(o::REGION, o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  //set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>(); 
  auto x_ps_tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  auto motion = ptcls->get<3>();
  
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      auto elmVerts = o::gather_verts<4>(cells2nodes, o::LO(e));
      auto vtxCoords = o::gather_vectors<4,3>(nodes2coords, elmVerts);
      const o::Real s = rand_nums[5*pid];
      const o::Real t = rand_nums[5*pid+1];
      const o::Real u = rand_nums[5*pid+2];
      const o::Real a = 1 - s - t - u;
      for (int i = 0; i < 3; i++) {
        x_ps_d(pid, i) = a * vtxCoords[0][i] + s * vtxCoords[1][i]
          + t * vtxCoords[2][i] + u * vtxCoords[3][i];
        x_ps_tgt(pid, i) = x_ps_d(pid, i);
      }
      const o::Real theta = rand_nums[5*pid + 3];
      const o::Real z = rand_nums[5*pid + 4];
      motion(pid, 0) = sqrt(1 - z*z) * cos(theta);
      motion(pid, 1) = sqrt(1 - z*z) * sin(theta);
      motion(pid, 2) = z;
      pids(pid) = pid;
    }
  };
  ps::parallel_for(ptcls, lamb);

}

template <int N>
o::Real determine_distance(o::Mesh mesh) {
  auto bb = o::get_bounding_box<N>(&mesh);
  char buffer[1024];
  char name[3] = {'x', 'y', 'z'};
  char* ptr = buffer + sprintf(buffer, "  Bounding Box:");
  for (int i = 0; i < N; ++i)
    ptr += sprintf(ptr, " |%c: [%f:%f]|", name[i], bb.min[i], bb.max[i]);
  fprintf(stderr, "%s\n", buffer);
  double maxDimLen = 0;
  for (int i = 0; i < N; ++i) {
    o::Real len = bb.max[i] - bb.min[i];
    if (len > maxDimLen)
      maxDimLen = len;
  }
  return maxDimLen;
}

//Initialize particle position and directions inside each element
void init_internal(o::Mesh mesh, PS* ptcls) {
  if (mesh.dim() == 2)
    init2DInternal(mesh, ptcls);
  else
    init3DInternal(mesh, ptcls);
}

void init_edges(o::Mesh mesh, PS* ptcls) {
  if (mesh.dim() == 2)
    init_2D_edges(mesh, ptcls);
  else
    init_3D_edges(mesh, ptcls);
}


o::Real get_push_distance(o::Mesh mesh) {
  if (mesh.dim() == 2)
    return determine_distance<2>(mesh) / (3*sqrt(mesh.nelems()));
  return determine_distance<3>(mesh) / (3*pow(mesh.nelems(), 1.0/3));
}

//Push particles
void push_ptcls(PS* ptcls, o::Real distance) {
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto angle = ptcls->get<3>();
  auto push = PS_LAMBDA(const p::lid_t elm, const p::lid_t ptcl, const bool mask) {
    if (mask) {
      for (int i = 0; i < 3; ++i) {
        tgt(ptcl, i) = tgt(ptcl,i) + distance * angle(ptcl, i);
      }
    }
  };
  p::parallel_for(ptcls, push, "push");
}

//Tests the particle is within its parent element and then sets the particle coordinates to the target position
bool test_parent_elements(o::Mesh mesh, PS* ptcls, o::Write<o::LO> elem_ids, const o::Real tol) {
  //Use pumipic routine in search to check element
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  o::Reals elmArea = measure_elements_real(&mesh);
  o::Write<o::LO> ptcl_done(ptcls->capacity(), 0, "ptcl_done");
  auto setDone = PS_LAMBDA(const o::LO, const o::LO ptcl, const bool mask) {
    ptcl_done[ptcl] = (!mask | (elem_ids[ptcl] == -1));
  };
  p::parallel_for(ptcls, setDone, "setDone");

  o::LO failures = p::check_initial_parents(mesh, ptcls, tgt, pids, elem_ids, ptcl_done, elmArea, true, tol);

  //Reset coordinates
  auto resetCoordinates = PS_LAMBDA(const o::LO elm, const o::LO ptcl, bool) {
    for (int i = 0; i < 3; ++i) {
      cur(ptcl, i) = tgt(ptcl, i);
    }
  };
  p::parallel_for(ptcls, resetCoordinates, "resetCoordinates");
  return failures;
}

//Checks particles that did not intersect the wall are within the mesh still
//Assumes the plate or cube mesh are used
template <int DIM>
bool check_inside_bbox(o::Mesh mesh, PS* ptcls, o::Write<o::LO> xFaces, const o::Real tol) {
  auto box = o::get_bounding_box<DIM>(&mesh);
  auto x_ps_tgt = ptcls->get<1>();
  o::Write<o::LO> failures(1,0);
  auto checkInteriors = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    const o::LO face = xFaces[ptcl];
    const o::Vector<3> ptcl_pos = makeVector3(ptcl, x_ps_tgt);
    if (mask && face == -1) {
      bool fail = false;
      for (int i = 0; i < DIM; ++i) {
        fail |= (ptcl_pos[i] < box.min[i] - tol);
        fail |= (ptcl_pos[i] > box.max[i] + tol);
      }
      Kokkos::atomic_add(&(failures[0]), (o::LO)fail);
    }
  };
  p::parallel_for(ptcls, checkInteriors, "checkInteriors");
  o::LO fails = o::HostWrite<o::LO>(failures)[0];
  if (fails != 0) {
    fprintf(stderr, "%d particles left the mesh without a model intersection calculated\n", fails);
  }
  return fails;
}

//Checks if the point is on an edge
OMEGA_H_INLINE bool check_point_on_edge(const o::Few<o::Vector<2>, 2> verts, const o::Vector<2> point, const o::Real tol, o::LO ptcl) {
  const o::Vector<2> path = point - verts[0];
  const o::Vector<2> edge = verts[1] - verts[0];
  const o::Real cross = o::cross(path, edge);

  return fabs(cross) <= tol &&
    point[0] >= Kokkos::min(verts[0][0], verts[1][0]) - tol &&
    point[1] >= Kokkos::min(verts[0][1], verts[1][1]) - tol &&
    point[0] <= Kokkos::max(verts[0][0], verts[1][0]) + tol &&
    point[1] <= Kokkos::max(verts[0][1], verts[1][1]) + tol;
}

//3D test of intersection point within the intersection face
bool check_intersections_3d(o::Mesh mesh, PS* ptcls, o::Write<o::LO> elem_ids,
                            o::Write<o::LO> intersection_faces, o::Write<o::Real> x_points, const o::Real tol) {
  o::Write<o::LO> failures(1,0);
  const auto face_verts =  mesh.ask_verts_of(2);
  const auto coords = mesh.coords();
  auto checkIntersections3d = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    const o::LO face = intersection_faces[ptcl];
    if (mask && face != -1) {
      //check intersection point is in the intersection face
      const auto fv2v = o::gather_verts<3>(face_verts, face);
      const auto abc = o::gather_vectors<3, 3>(coords, fv2v);
      Omega_h::Vector<3> xpoint = o::zero_vector<3>();
      for (int i =0; i < 3; ++i)
        xpoint[i] = x_points[3 * ptcl + i];
      Omega_h::Vector<3> bcc;
      bool pass = p::find_barycentric_tri_simple(abc, xpoint, bcc);
      if(pass && !p::all_positive(bcc, tol)) {
        Kokkos::atomic_add(&(failures[0]),1);
        printf("[ERROR] Particle intersected with model boundary outside the intersection face!\n"
               "  Initial Element: %d  New Element %d\n  BCC: %.15f %.15f %.15f\n", elm, elem_ids[ptcl],
               bcc[0], bcc[1], bcc[2]);
      }
    }
  };
  p::parallel_for(ptcls, checkIntersections3d, "checkIntersections3d");
  bool fail = o::HostWrite<o::LO>(failures)[0];
  return fail;
}

//2D test of intersection point within the intersection edge
bool check_intersections_2d(o::Mesh mesh, PS* ptcls, o::Write<o::LO> intersection_faces, o::Write<o::Real> x_points, const o::Real tol) {
  o::Write<o::LO> failures(1,0);
  const auto edge_verts =  mesh.ask_verts_of(1);
  const auto coords = mesh.coords();
  auto checkIntersections2d = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    const o::LO edge = intersection_faces[ptcl];
    if (mask && edge != -1) {
      //check intersection point is in the intersection face
      const auto ev2v = o::gather_verts<2>(edge_verts, edge);
      const auto abc = o::gather_vectors<2, 2>(coords, ev2v);
      Omega_h::Vector<2> xpoint = o::zero_vector<2>();
      for (int i =0; i < 2; ++i)
        xpoint[i] = x_points[2 * ptcl + i];
      if (!check_point_on_edge(abc, xpoint, tol, ptcl)) {
        Kokkos::atomic_add(&(failures[0]),1);
        printf("[ERROR] Particle intersected with model boundary outside the intersection edge!\n");
      }
    }
  };
  p::parallel_for(ptcls, checkIntersections2d, "checkIntersections2d");
  bool fail = o::HostWrite<o::LO>(failures)[0];
  return fail; 
}

//Tests particles that hit the wall if the intersection points are correct/valid
//Also calls check_inside_bbox to check particles that did not find a wall intersection to ensure they are in the mesh
template <int DIM>
bool test_wall_intersections(o::Mesh mesh, PS* ptcls, o::Write<o::LO> elem_ids, o::Write<o::LO> intersection_faces,
                             o::Write<o::Real> x_points, const o::Real tol) {
  bool fail = false;
  o::Write<o::LO> failures(1,0);
  auto bb = o::get_bounding_box<DIM>(&mesh);
  int nvpe = DIM + 1;
  const auto elm_down = mesh.ask_down(mesh.dim(), mesh.dim() - 1);
  auto motion = ptcls->get<3>();
  //Test intersection points on face
  if (DIM == 3)
    fail |= check_intersections_3d(mesh, ptcls, elem_ids, intersection_faces, x_points, tol);
  else if (DIM == 2)
    fail |= check_intersections_2d(mesh, ptcls, intersection_faces, x_points, tol);

  //Test intersection points against motion and intersection face on parent element
  auto x_ps_orig = ptcls->get<0>();
  auto x_ps_tgt = ptcls->get<1>();
  auto checkIntersections = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    const o::LO face = intersection_faces[ptcl];
    const o::LO searchElm = elem_ids[ptcl];
    if (mask && face != -1) {
      //Check the intersection point is along the particle path [norm(C-A) = norm(B-A)]
      Omega_h::Vector<DIM> xpoint = o::zero_vector<DIM>();
      Omega_h::Vector<DIM> orig = o::zero_vector<DIM>();
      Omega_h::Vector<DIM> tgt = o::zero_vector<DIM>();
      for (int i =0; i < DIM; ++i) {
        orig[i] = x_ps_orig(ptcl, i);
        tgt[i] = x_ps_tgt(ptcl, i);
        xpoint[i] = x_points[DIM * ptcl + i];
      }
      const o::Vector<DIM> dir = o::normalize(tgt - orig);
      const o::Vector<DIM> dir2 = o::normalize(xpoint - orig);
      for (int i = 0; i < DIM; ++i) {
        if (fabs(dir[i] - dir2[i]) > tol && dir[i] > tol && dir2[i] > tol) {
          Kokkos::atomic_add(&(failures[0]), 1);
          printf("[ERROR] Intersection point is not along the particle path\n"
                 "  dir: %.15f %.15f %.15f\n  dir2 %.15f %.15f %.15f\n", dir[0], dir[1], dir[2], dir2[0], dir2[1], dir2[2]);
        }
      }
      
      //check new parent element has the intersection face
      bool hasFace = false;
      for (int i =0; i < nvpe; ++i) {
        hasFace |= (elm_down.ab2b[nvpe*searchElm + i] == face);
      }
      if (!hasFace) {
        Kokkos::atomic_add(&(failures[0]),1);
        printf("[ERROR] Intersection face is not adjacent to parent element!\n");
      }
    }
  };
  p::parallel_for(ptcls, checkIntersections, "checkIntersections");

  fail |= o::HostWrite<o::LO>(failures)[0];
  //check non intersected particles are still in the domain
  fail |= check_inside_bbox<DIM>(mesh, ptcls, intersection_faces, tol);
  return fail;
}

bool testBCCSearch(o::Mesh mesh, PS* ptcls, const o::Real tol) {
  bool fail = false;
  o::Real distance = get_push_distance(mesh);
  
  //Push particles
  fprintf(stderr, "  First push\n");
  push_ptcls(ptcls, distance);

  //Search
  fprintf(stderr, "  Perform search\n");
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  o::Write<o::LO> elem_ids;
  o::Write<o::LO> xFaces;
  o::Write<o::Real> xPoints;
  fail |= !p::search_mesh(mesh, ptcls, cur, tgt, pids, elem_ids, false, xFaces, xPoints);

  //Test search worked
  fprintf(stderr, "  Testing resulting parent elements\n");
  fail |= test_parent_elements(mesh, ptcls, elem_ids, tol);


  //Push particles again
  fprintf(stderr, "  Push particles again\n");
  push_ptcls(ptcls, distance);

  //Search again
  fprintf(stderr, "  Find second new parent elements\n");
  fail |= !p::search_mesh(mesh, ptcls, cur, tgt, pids, elem_ids, false, xFaces, xPoints);

  //Test search again
  fprintf(stderr, "  Testing resulting parent elements again\n");
  fail |= test_parent_elements(mesh, ptcls, elem_ids, tol);
  return fail;
}

bool test_internal_BCC_search(o::Mesh mesh, PS* ptcls, const o::Real tol) {
  fprintf(stderr, "\nBeginning BCC Search test with internal particle positions\n");
  //Initialize particle information
  init_internal(mesh, ptcls);

  //Run test
  return testBCCSearch(mesh, ptcls, tol);
}

bool test_edge_BCC_search(o::Mesh mesh, PS* ptcls, const o::Real tol) {
  fprintf(stderr, "\nBeginning BCC Search test with edge particle positions\n");
  //Initialize particle information
  init_edges(mesh, ptcls);

  //Run test
  return testBCCSearch(mesh, ptcls, tol);
}

void reflect_intersections(PS* ptcls, o::Write<o::LO> xFaces, o::Write<o::Real> xPoints) {
  auto tgt = ptcls->get<1>();
  auto dir = ptcls->get<3>();
  int dim = xPoints.size() / ptcls->capacity();
  auto reflectParticles = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    o::LO face = xFaces[ptcl];
    if (mask && face != -1) {
      for (int i = 0; i < dim; ++i) {
        tgt(ptcl, i) =  xPoints[ptcl*dim + i];
        dir(ptcl, i) *= -1;
      }
    }
  };
  p::parallel_for(ptcls, reflectParticles, "reflectParticles");
}

void delete_intersections(PS* ptcls, o::Write<o::LO> elem_ids, o::Write<o::LO> xFaces) {
  auto deleteIntersections = PS_LAMBDA(const o::LO elm, const o::LO ptcl, const bool mask) {
    o::LO face = xFaces[ptcl];
    if (mask && face != -1) {
      elem_ids[ptcl] = -1;
    }
  };
  p::parallel_for(ptcls, deleteIntersections, "deleteIntersections");

}

bool test_intersection_search(o::Mesh mesh, PS* ptcls, const o::Real tol) {
  bool fail = false;
  o::Real distance = get_push_distance(mesh);

    //Push particles
  fprintf(stderr, "  First push\n");
  for (int i = 0; i < 10; ++i)
    push_ptcls(ptcls, distance);

  //Search
  fprintf(stderr, "  Perform search\n");
  auto cur = ptcls->get<0>();
  auto tgt = ptcls->get<1>();
  auto pids = ptcls->get<2>();
  o::Write<o::LO> elem_ids;
  o::Write<o::LO> xFaces;
  o::Write<o::Real> xPoints;
  fail |= !p::search_mesh(mesh, ptcls, cur, tgt, pids, elem_ids, 
                          true, xFaces, xPoints, 0, true);

  //Test wall intersections
  fprintf(stderr, "  Testing wall intersections\n");
  if (mesh.dim() == 2)
    fail |= test_wall_intersections<2>(mesh, ptcls, elem_ids, xFaces, xPoints, tol);
  else
    fail |= test_wall_intersections<3>(mesh, ptcls, elem_ids, xFaces, xPoints, tol);

  //Deal with intersections by reflection
  reflect_intersections(ptcls, xFaces, xPoints);
  
  //Test search worked
  fprintf(stderr, "  Testing resulting parent elements\n");
  fail |= test_parent_elements(mesh, ptcls, elem_ids, tol);

  //Push particles again
  fprintf(stderr, "  Push particles again\n");
  push_ptcls(ptcls, distance);

  //Search again
  fprintf(stderr, "  Find second new parent elements\n");
  fail |= !p::search_mesh(mesh, ptcls, cur, tgt, pids, elem_ids, 
                          true, xFaces, xPoints, 0, true);

  //Test wall intersections
  fprintf(stderr, "  Testing wall intersections\n");
  if (mesh.dim() == 2)
    fail |= test_wall_intersections<2>(mesh, ptcls, elem_ids, xFaces, xPoints, tol);
  else
    fail |= test_wall_intersections<3>(mesh, ptcls, elem_ids, xFaces, xPoints, tol);

  //Deal with intersections by deletion
  delete_intersections(ptcls, elem_ids, xFaces);

  //Test search again
  fprintf(stderr, "  Testing resulting parent elements again\n");
  fail |= test_parent_elements(mesh, ptcls, elem_ids, tol);

  return fail;
}

bool test_internal_intersection_search(o::Mesh mesh, PS* ptcls, const o::Real tol) {
  fprintf(stderr, "\nBeginning search with intersection points test with internal particle positions\n");
  //Initialize particle information
  init_internal(mesh, ptcls);
  return test_intersection_search(mesh, ptcls, tol);
}

bool test_edge_intersection_search(o::Mesh mesh, PS* ptcls, const o::Real tol) {
  fprintf(stderr, "\nBeginning search with intersection points test with edge particle positions\n");
  //Initialize particle information
  init_edges(mesh, ptcls);
  return test_intersection_search(mesh, ptcls, tol);
}

int test_search(o::Mesh mesh, p::lid_t np, const o::Real tol) {
  
  fprintf(stderr, "\n%s\n", std::string(20, '-').c_str());
  fprintf(stderr, "Begin testing with %d particles\n", np);
  
  //Create particle structure
  PS* ptcls = create_particle_structure(mesh, np);

  //Run tests
  int fails = 0;
  fails += test_internal_BCC_search(mesh, ptcls, tol);
  fails += test_internal_intersection_search(mesh, ptcls, tol);
  fails += test_edge_BCC_search(mesh, ptcls, tol);
  fails += test_edge_intersection_search(mesh, ptcls, tol);

  delete ptcls;
  return fails;
}

int main(int argc, char** argv) {

  if(argc != 2)
  {
    fprintf(stderr, "Usage: %s <mesh>\n", argv[0]);
    return 1;
  }

  //Initialize Omega_h and read in mesh
  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  Omega_h::Mesh mesh = Omega_h::read_mesh_file(argv[1], lib.world());

  o::Reals elmArea = measure_elements_real(&mesh);
  o::Real tol = p::compute_tolerance_from_area(elmArea);

  int fails = 0;
  fails += test_search(mesh, 100, tol);
#ifdef PP_USE_CUDA
  fails += test_search(mesh, 1000000, tol);
#else
  fails += test_search(mesh, 10000, tol);
#endif

  if (fails == 0) {
    fprintf(stderr, "\nAll Tests Passed\n");
  }
  return fails;
}
