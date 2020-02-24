#include "xgcm_gyro_scatter.hpp"
#include <pumipic_adjacency.hpp>
#include <Omega_h_for.hpp>

namespace xgcm {
  using GyroField=Mesh::GyroField;
  using GyroFieldR=Mesh::GyroFieldR;

  namespace {
    o::Real gyro_rmax = 0.038; //max ring radius
    o::LO gyro_num_rings = 3;
    o::LO gyro_points_per_ring = 8;
    o::Real gyro_theta = 0;

    typedef ps::MemberTypes<Vector3d, Vector3d, int> Point;
    typedef ps::ParticleStructure<Point> PSpt;

  }

  void setGyroConfig(Input& input) {
    gyro_rmax = input.gyro_rmax;
    gyro_num_rings = input.gyro_num_rings;
    gyro_points_per_ring = input.gyro_points_per_ring;
    gyro_theta = input.gyro_theta;
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
    Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
    PSpt::kkGidView empty_gids("empty_gids", 0);
    PSpt* gyro_ps = new ps::SellCSigma<Point>(policy, sigma, V,
                                              mesh->nelems(), num_points,
                                              ptcls_per_elem, empty_gids,
                                              point_element, point_info);
    printf("created ps for gyro mapping with %d points and %d elms\n",
           num_points, mesh->nelems());

    //Adjacency search
    auto start = gyro_ps->get<PTCL_COORDS>();
    auto end = gyro_ps->get<PTCL_TARGET>();
    auto pids = gyro_ps->get<PTCL_IDS>();
    int maxLoops = 100;
    int psCapacity = gyro_ps->capacity();
    o::Write<o::LO> elem_ids(psCapacity, -1);
    bool isFound = p::search_mesh_2d(*mesh, gyro_ps, start, end, pids,
                                     elem_ids, maxLoops);
    assert(isFound);

    const auto numElms = mesh->nelems();
    //Gyro avg mapping: 3 vertices per ring point (Assumes all elements are triangles)
    const o::LO nvpe = 3;
    o::Write<o::LO> gyro_avg_map(nvpe * num_points, -1, "gyro_map");
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
  void createIonGyroRingMappings(o::Mesh* mesh, o::LOs& major_map,
                              o::LOs& minor_map) {
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
    o::Write<o::Real> major_ring_points(num_points * 2);
    o::Write<o::Real> minor_ring_points(num_points * 2);
    auto projectCoords = OMEGA_H_LAMBDA(const o::LO& id) {
      //TODO Project the points along field lines
      for (int i = 0; i < 2; ++i) {
        major_ring_points[id*2+i] = ring_points[id*2+i];
        minor_ring_points[id*2+i] = ring_points[id*2+i];
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
    major_map = searchAndBuildMap(mesh, o::Reals(element_centroids),
                                    o::Reals(major_ring_points),
                                    o::LOs(starting_element));
    minor_map = searchAndBuildMap(mesh, o::Reals(element_centroids),
                                     o::Reals(minor_ring_points),
                                     o::LOs(starting_element));
    Kokkos::Profiling::popRegion();
  }

  void gyroScatter(Mesh& mesh, PS_I* ptcls, o::LOs v2v, GyroField scatter_w) {
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
    ps::parallel_for(ptcls, accumulateToRings);
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
    if(!rank || rank == comm_size/2)
      fprintf(stderr, "%d gyro scatter (seconds) %f pre-barrier (seconds) %f\n",
              rank, timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

  void gyroSync(Mesh& m, GyroField major, GyroField minor) {
    const auto btime = pumipic_prebarrier();
    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion("xgcm_gyroSync");
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    Kokkos::Timer gathertimer;
    m.gatherField(0, major, GyroFieldR(minor));
    const auto gtime = gathertimer.seconds();
    /*
       Note: In the future, a solve will happen between the gather and scatter field calls.
             For now we will do one after the other to essentially perform an allreduce
    */
    Kokkos::Timer scattertimer;
    m.scatterField(0, major, minor);
    const auto stime = scattertimer.seconds();
    if(!rank || rank == comm_size/2) {
      fprintf(stderr, "%d gyro sync times (sec): sync %f pre-barrier %f gather %f "
              "scatter %f\n",
              rank, timer.seconds(), btime, gtime, stime);
    }
    Kokkos::Profiling::popRegion();
  }

  void gyroScatter(Mesh& mesh, PS_I* ptcls) {
    GyroField major, minor;
    mesh.getGyroFields(major,minor);
    Omega_h::LOs major_map, minor_map;
    mesh.getIonGyroMappings(major_map, minor_map);
    gyroScatter(mesh, ptcls, major_map, major);
    gyroScatter(mesh, ptcls, minor_map, minor);
    gyroSync(mesh, major, minor);
  }

  void gyroScatter(Mesh& mesh, PS_E* ptcls) {

  }
}
