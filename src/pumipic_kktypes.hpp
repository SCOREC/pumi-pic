#ifndef PUMIPIC_KKTYPES_H
#define PUMIPIC_KKTYPES_H

#include <Kokkos_Core.hpp>
#include <psTypes.h>
#include <chrono>
#include <thread>

namespace pumipic {

using particle_structs::fp_t;
using particle_structs::lid_t;
using particle_structs::Vector3d;

typedef Kokkos::DefaultExecutionSpace exe_space;
typedef Kokkos::View<lid_t*, exe_space::device_type> kkLidView;
inline void hostToDeviceLid(kkLidView d, lid_t *h) {
  kkLidView::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size(); ++i) {
    hv(i) = h[i];
  }
  Kokkos::deep_copy(d,hv);
}

inline void deviceToHostLid(kkLidView d, lid_t *h) {
  kkLidView::HostMirror hv = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(hv,d);
  for(size_t i=0; i<hv.size(); ++i) {
    h[i] = hv(i);
  }
}

typedef Kokkos::View<fp_t*, exe_space::device_type> kkFpView;
/** \brief helper function to transfer a host array to a device view */
inline void hostToDeviceFp(kkFpView d, fp_t* h) {
  kkFpView::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size(); ++i)
    hv(i) = h[i];
  Kokkos::deep_copy(d,hv);
}

typedef Kokkos::View<Vector3d*, exe_space::device_type> kkFp3View;
/** \brief helper function to transfer a host array to a device view */
inline void hostToDeviceFp(kkFp3View d, fp_t (*h)[3]) {
  kkFp3View::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size()/3; ++i) {
    hv(i,0) = h[i][0];
    hv(i,1) = h[i][1];
    hv(i,2) = h[i][2];
  }
  Kokkos::deep_copy(d,hv);
}

inline void deviceToHostFp(kkFp3View d, fp_t (*h)[3]) {
  kkFp3View::HostMirror hv = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(hv,d);
  for(size_t i=0; i<hv.size()/3; ++i) {
    h[i][0] = hv(i,0);
    h[i][1] = hv(i,1);
    h[i][2] = hv(i,2);
  }
}

inline void deviceToHostFp(kkFpView d, fp_t *h) {
  kkFpView::HostMirror hv = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(hv,d);
  for(size_t i=0; i<hv.size(); ++i) {
    h[i] = hv(i);
  }
}

}
#endif
