#include <cmath>
#include <cstdio>
#include "pseudoXGCmTypes.hpp"
#include "pumipic_profiling.hpp"

namespace ellipticalPush {
  double h; //x coordinate of center
  double k; //y coordinate of center
  double d; //ratio of ellipse minor axis length (a) to major axis length (b)

  void setup(SCS* scs, const double h_in, const double k_in,
      const double d_in) {
    h = h_in;
    k = k_in;
    d = d_in;
    auto x_nm1 = scs->get<0>();
    auto ptcl_id = scs->get<2>();
    auto ptcl_b = scs->get<3>();
    auto ptcl_phi = scs->get<4>();
    const auto h_d = h;
    const auto k_d = k;
    const auto d_d = d;
    auto setMajorAxis = SCS_LAMBDA(const int&, const int& pid, const int& mask) {
      if(mask) {
        const auto w = x_nm1(pid,0);
        const auto z = x_nm1(pid,1);
        const auto v = std::sqrt(std::pow(w-h_d,2) + std::pow(z-k_d,2));
        const auto phi = atan2(d_d*(z-k_d),w-h_d);
        const auto b = (z - k_d)/sin(phi);
        ptcl_phi(pid) = phi;
        ptcl_b(pid) = b;
      }
    };
    scs->parallel_for(setMajorAxis);
  }
  
  void push(SCS* scs, Omega_h::Mesh& m, const double deg, const int iter) {
    const auto btime = pumipic_prebarrier();
    Kokkos::Profiling::pushRegion("ellipticalPush");
    Kokkos::Timer timer;
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    auto class_ids = m.get_array<Omega_h::ClassId>(m.dim(), "class_id");
    auto x_nm0 = scs->get<1>();
    auto ptcl_id = scs->get<2>();
    auto ptcl_b = scs->get<3>();
    auto ptcl_phi = scs->get<4>();
    const auto h_d = h;
    const auto k_d = k;
    const auto d_d = d;
    auto setPosition = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if(mask) {
        const double centerFactor = class_ids[e] == 1 ? 0.01 : 1.0;
        const double distByClass = centerFactor * (double) 1.0 / class_ids[e];
        const auto degP = deg*distByClass;
        const auto phi = ptcl_phi(pid);
        const auto b = ptcl_b(pid);
        const auto a = b*d_d;
        const auto rad = phi+degP*M_PI/180.0;
        const auto x = a*std::cos(rad)+h_d;
        const auto y = b*std::sin(rad)+k_d;
        x_nm0(pid,0) = x;
        x_nm0(pid,1) = y;
        ptcl_phi(pid) = rad;
      }
    };
    scs->parallel_for(setPosition);
    if(!rank || rank == comm_size/2)
      fprintf(stderr, "%d elliptical push (seconds) %f pre-barrier (seconds) %f\n",
          rank, timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }
}
