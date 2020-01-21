#include "xgcp_push.hpp"
#include <PS_Macros.h>
#include "pumipic_profiling.hpp"

namespace xgcp {
  namespace ellipticalPush {
    double h,k,d;
    void setup(PS_I* ptcls, const double h_, const double k_, const double d_) {
      h = h_;
      k = k_;
      d = d_;
      auto x_nm1 = ptcls->get<0>();
      auto ptcl_id = ptcls->get<2>();
      auto ptcl_b = ptcls->get<3>();
      auto ptcl_phi = ptcls->get<4>();
      const auto h_d = h;
      const auto k_d = k;
      const auto d_d = d;
      auto setMajorAxis = PS_LAMBDA(const int&, const int& pid, const int& mask) {
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
      ps::parallel_for(ptcls, setMajorAxis);
    }
    void push(PS_I* ptcls, Omega_h::Mesh& m, const double deg, const int iter) {
      const auto btime = pumipic_prebarrier();
      Kokkos::Profiling::pushRegion("ellipticalPush");
      Kokkos::Timer timer;
      int rank, comm_size;
      MPI_Comm_rank(MPI_COMM_WORLD,&rank);
      MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
      auto class_ids = m.get_array<Omega_h::ClassId>(m.dim(), "class_id");
      auto x_c = ptcls->get<0>();
      auto x_nm0 = ptcls->get<1>();
      auto ptcl_id = ptcls->get<2>();
      auto ptcl_b = ptcls->get<3>();
      auto ptcl_phi = ptcls->get<4>();
      const auto h_d = h;
      const auto k_d = k;
      const auto d_d = d;
      auto setPosition = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
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
          x_nm0(pid,2) = x_c(pid, 2) + degP * M_PI/180.0;
          x_nm0(pid,2) -= (x_nm0(pid,2) > M_PI * 2) * M_PI * 2;
          ptcl_phi(pid) = rad;
        }
      };
      ps::parallel_for(ptcls, setPosition);
      if(!rank || rank == comm_size/2)
        fprintf(stderr, "%d elliptical push (seconds) %f pre-barrier (seconds) %f\n",
                rank, timer.seconds(), btime);
      Kokkos::Profiling::popRegion();
    }
  }
}
