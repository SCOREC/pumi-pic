#ifndef GYRO_SCATTER_H
#define GYRO_SCATTER_H

#include "pseudoXGCmTypes.hpp"

namespace {
  o::Real gyro_rmax = 0.038; //max ring radius
  o::LO gyro_num_rings = 3;
  o::LO gyro_points_per_ring = 8;
  o::Real gyro_theta = 0;
}

void setGyroConfig(o::Real rmax, o::LO nrings, o::LO pointsPerRing, o::LO theta) {
  gyro_rmax = rmax;
  gyro_num_rings = nrings;
  gyro_points_per_ring = pointsPerRing;
  gyro_theta = theta;
}

void printGyroConfig() {
  printf("gyro rmax num_rings points_per_ring theta %f %d %d %f\n",
      gyro_rmax, gyro_num_rings, gyro_points_per_ring, gyro_theta);
}

o::LOs searchAndBuildMap(o::Mesh* mesh, o::Reals element_centroids,
                               o::Reals projected_points, o::LOs starting_element) {
  o::LO num_points = starting_element.size();

  //Create PS for the projected points to perform adjacency search on
  PSpt::kkLidView ptcls_per_elem("ptcls_per_elem", mesh->nelems());
  PSpt::kkLidView point_element("point_element", num_points);
  auto point_info = particle_structs::createMemberViews<Point>(num_points);
  auto start_pos = particle_structs::getMemberView<Point, 0>(point_info);
  auto end_pos = particle_structs::getMemberView<Point, 1>(point_info);
  auto point_id = particle_structs::getMemberView<Point, 2>(point_info);
  auto countPointsInElement = OMEGA_H_LAMBDA(const o::LO& id) {
    const o::LO elm = starting_element[id];
    Kokkos::atomic_fetch_add(&(ptcls_per_elem(elm)), 1); //can just be 'add', no fetch needed
    point_id(id) = id;
    point_element(id) = elm;
    for (int i = 0; i < 2; ++i) {
      start_pos(id,i) = element_centroids[elm*2+i];
      end_pos(id,i) = projected_points[id*2+i];
    }
  };
  o::parallel_for(num_points, countPointsInElement, "countPointsInElement");

  const int sigma = INT_MAX;
  const int V = 64;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy = TeamPolicyAuto(10000, 32);
  PSpt::kkGidView empty_gids("empty_gids", 0);
  PSpt* gyro_ps = new ps::SellCSigma<Point>(policy, sigma, V,
                                            mesh->nelems(), num_points,
                                            ptcls_per_elem, empty_gids,
                                            point_element, point_info);
  printf("created ps for gyro mapping with %d points and %d elms\n",
      num_points, mesh->nelems());

  //Adjacency search
  auto start = gyro_ps->get<0>();
  auto end = gyro_ps->get<1>();
  auto pids = gyro_ps->get<2>();
  int maxLoops = 100;
  int psCapacity = gyro_ps->capacity();
  o::Write<o::LO> elem_ids(psCapacity, -1);
  bool isFound = p::search_mesh_2d(*mesh, gyro_ps, start, end, pids,
                                   elem_ids, maxLoops);
  assert(isFound);

  const auto numElms = mesh->nelems();
  //Gyro avg mapping: 3 vertices per ring point (Assumes all elements are triangles)
  const o::LO nvpe = 3;
  o::Write<o::LO> gyro_avg_map(nvpe * num_points, -1);
  auto elm2Verts = mesh->ask_down(mesh->dim(), 0);
  auto createGyroMapping = PS_LAMBDA(const int&, const int& pid, const int& mask) {
    const o::LO parent = elem_ids[pid];
    if (mask && parent >= 0) { //skip points outside the domain (parent == -1)
      assert(parent>=0 && parent<numElms);
      const o::LO id = pids(pid);
      const o::LO start_index = id* nvpe;
      const o::LO start_elm = parent*nvpe;
      for (int i = 0; i < 3; ++i)
        gyro_avg_map[start_index+i] = elm2Verts.ab2b[start_elm+i];
    }
  };
  ps::parallel_for(gyro_ps, createGyroMapping);
  delete gyro_ps;
  return o::LOs(gyro_avg_map);
}




/* Build gyro-avg mapping */
void createGyroRingMappings(o::Mesh* mesh, o::LOs& forward_map,
                           o::LOs& backward_map) {
  Kokkos::Profiling::pushRegion("xgcm_createGyroRingMappings");
  const auto gr = gyro_rmax;
  const auto gnr = gyro_num_rings;
  const auto gppr = gyro_points_per_ring;
  const auto gt = gyro_theta; //degrees to offset ring point distribution
  o::LO nverts = mesh->nverts();
  o::LO num_points = nverts * gnr * gppr;
  o::Write<o::Real> ring_points(num_points * 2);
  o::Reals coords = mesh->coords();
  const auto torad = M_PI/180;

  //Calculate points of each ring around every vertex
  auto generateRingPoints = OMEGA_H_LAMBDA(const o::LO& id) {
    const o::LO point_id = id %  gppr;
    const o::LO id2 = id /gppr;
    const o::LO ring_id = id2 % gnr;
    const o::LO vert_id = id2 / gnr;
    const o::Real radius = gr*(ring_id+1)/gnr;
    const o::Real deg = gt + (((o::Real)point_id)/gppr * 360);
    const o::Real rad = deg * torad;
    ring_points[id*2] = coords[vert_id*2] + radius * cos(rad);
    ring_points[id*2+1] = coords[vert_id*2+1] + radius * sin(rad);
  };
  o::parallel_for(num_points, generateRingPoints, "generateRingPoints");

  //Project ring points to +/- planes
  o::Write<o::Real> forward_ring_points(num_points * 2);
  o::Write<o::Real> backward_ring_points(num_points * 2);
  auto projectCoords = OMEGA_H_LAMBDA(const o::LO& id) {
    //TODO Project the points along field lines
    for (int i = 0; i < 2; ++i) {
      forward_ring_points[id*2+i] = ring_points[id*2+i];
      backward_ring_points[id*2+i] = ring_points[id*2+i];
    }
  };
  o::parallel_for(num_points, projectCoords, "projectCoords");

  //Use adjacency search to find the element that the projected point is in
  o::Write<o::LO> starting_element(num_points,0, "starting_element");
  auto verts2Elm = mesh->ask_up(0, mesh->dim());
  auto setInitialElement = OMEGA_H_LAMBDA(const o::LO& id) {
    //Note: Setting initial element to be the first element adjacent to the vertex
    const o::LO vert_id = id / gppr / gnr;
    const auto firstElm = verts2Elm.a2ab[vert_id];
    starting_element[id] = verts2Elm.ab2b[firstElm];
  };
  o::parallel_for(num_points, setInitialElement, "setInitialElement");

  //Calculate centroids of each element
  o::Write<o::Real> element_centroids(mesh->nelems()*2);
  auto cells2nodes = mesh->get_adj(mesh->dim(), o::VERT).ab2b;
  auto calculateCentroids = OMEGA_H_LAMBDA(const o::LO& elm) {
    auto elmVerts = o::gather_verts<3>(cells2nodes, elm);
    auto vtxCoords = o::gather_vectors<3,2>(coords, elmVerts);
    auto center = average(vtxCoords);
    element_centroids[elm*2] = center[0];
    element_centroids[elm*2+1] = center[1];
  };
  o::parallel_for(mesh->nelems(), calculateCentroids, "calculateCentroids");

  //Create both mapping
  forward_map = searchAndBuildMap(mesh, o::Reals(element_centroids),
                                  o::Reals(forward_ring_points),
                                  o::LOs(starting_element));
  backward_map = searchAndBuildMap(mesh, o::Reals(element_centroids),
                                   o::Reals(backward_ring_points),
                                   o::LOs(starting_element));
  Kokkos::Profiling::popRegion();
}

void gyroScatter(o::Mesh* mesh, PS* ptcls, o::LOs v2v, std::string scatterTagName) {
  const auto btime = pumipic_prebarrier();
  Kokkos::Timer timer;
  Kokkos::Profiling::pushRegion("xgcm_gyroScatter");
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  const auto gr = gyro_rmax;
  const auto gnr = gyro_num_rings;
  const auto gppr = gyro_points_per_ring;
  auto elm2verts = mesh->ask_down(mesh->dim(), 0);
  auto nvpe = 3; //triangles
  const double ringWidth = gr/gnr;
  o::Write<o::Real> ring_accum(gnr*mesh->nverts(),0, "ring_accumulator");
  auto accumulateToRings = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask > 0) {
      const auto ptclRadius = ringWidth*1.125; //TODO compute the radius
      assert(ptclRadius >= ringWidth);
      auto ringDown = 0;
      for(int i=2; i<=gnr; i++)
        ringDown += (ptclRadius >= ringWidth*i);
      auto ringUp = ringDown+1;
      assert(ringUp<gnr);
      assert(ptclRadius >= ringWidth*(ringDown+1));
      assert(ptclRadius < ringWidth*(ringUp+1));
      const auto firstVtx = e*nvpe;
      for(int i=0; i<nvpe; i++) {
        const auto v = elm2verts.ab2b[firstVtx+i];
        const auto vtxIdx = v*gnr;
        const auto ringUpIdx = vtxIdx+ringUp;
        const auto ringDownIdx = vtxIdx+ringDown;
        Kokkos::atomic_fetch_add(&(ring_accum[ringUpIdx]), 1); //replace with add
        Kokkos::atomic_fetch_add(&(ring_accum[ringDownIdx]), 1); //replace with add
      }
    }
  };
  ps::parallel_for(ptcls,accumulateToRings);
  const Omega_h::LO nverts = mesh->nverts();
  o::Write<o::Real> scatter_w(mesh->nverts(),0,"scatterTag_w");
  auto scatterToMappedVerts = OMEGA_H_LAMBDA(const o::LO& v) {
     const auto vtxIdx = v*gnr*gppr;
     const auto gyroVtxIdx = v*gnr;
     for(int ring=0; ring<gnr; ring++) {
        const auto accumRingVal = ring_accum[gyroVtxIdx+ring]/gppr;
        const auto ringIdx = ring*gppr;
        for(int pt=0; pt<gppr; pt++) {
          const auto ptIdx = 3*(vtxIdx+ringIdx+pt);
          for(int elmVtx=0; elmVtx<nvpe; elmVtx++) {
            const auto mappedIdx = ptIdx + elmVtx;
            const auto mappedVtx = v2v[mappedIdx];
            if (mappedVtx >= 0)
              Kokkos::atomic_fetch_add(&(scatter_w[mappedVtx]), accumRingVal); //replace with add
          }
        }
     }
  };
  o::parallel_for(mesh->nverts(), scatterToMappedVerts, "xgcm_scatterToMappedVerts");
  mesh->set_tag(o::VERT, scatterTagName, o::Reals(scatter_w));

  pumipic::RecordTime("gyro scatter", timer.seconds(), btime);
  Kokkos::Profiling::popRegion();
}

void gyroSync(p::Mesh& picparts, const std::string& fwdTagName,
              const std::string& bkwdTagName, const std::string& syncTagName) {
  const auto btime = pumipic_prebarrier();
  Kokkos::Timer timer;
  Kokkos::Profiling::pushRegion("xgcm_gyroSync");
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  Omega_h::Write<Omega_h::Real> sync_array = picparts.createCommArray(0, 2, Omega_h::Real(0.0));
  Omega_h::Mesh* mesh = picparts.mesh();
  Omega_h::Read<Omega_h::Real> fwdTag = mesh->get_array<Omega_h::Real>(0, fwdTagName);
  Omega_h::Read<Omega_h::Real> bkwdTag = mesh->get_array<Omega_h::Real>(0, bkwdTagName);

  auto setSyncArray = OMEGA_H_LAMBDA(const Omega_h::LO vtx_id) {
    sync_array[2*vtx_id] = fwdTag[vtx_id];
    sync_array[2*vtx_id+1] = bkwdTag[vtx_id];
  };
  Omega_h::parallel_for(mesh->nverts(), setSyncArray);

  Kokkos::Timer reducetimer;
  picparts.reduceCommArray(0, p::Mesh::Op::SUM_OP, sync_array);
  const auto rtime = reducetimer.seconds();

  mesh->set_tag(0, syncTagName, Omega_h::Reals(sync_array));
  pumipic::RecordTime("gyro sync", timer.seconds(), btime);
  pumipic::RecordTime("gyro reduction", rtime);
  Kokkos::Profiling::popRegion();
}
#endif
