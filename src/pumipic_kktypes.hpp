#ifndef PUMIPIC_KKTYPES_H
#define PUMIPIC_KKTYPES_H

#include <Kokkos_Core.hpp>
#include <SellCSigma.h>
#include <psTypes.h>
#include <chrono>
#include <thread>

namespace pumipic {
  using particle_structs::fp_t;
  using particle_structs::lid_t;
  using particle_structs::Vector3d;

  typedef Kokkos::DefaultExecutionSpace exe_space;
  typedef exe_space::device_type device_type;
  typedef particle_structs::Segment<int, device_type> SegmentInt;
  typedef particle_structs::Segment<Vector3d, device_type> Segment3d;

  typedef Kokkos::View<lid_t*, device_type> kkLidView;
  void hostToDeviceLid(kkLidView d, lid_t *h);
  void deviceToHostLid(kkLidView d, lid_t *h);
  typedef Kokkos::View<fp_t*, device_type> kkFpView;
  /** \brief helper function to transfer a host array to a device view */
  void hostToDeviceFp(kkFpView d, fp_t* h);
  typedef Kokkos::View<Vector3d*, device_type> kkFp3View;
  /** \brief helper function to transfer a host array to a device view */
  void hostToDeviceFp(kkFp3View d, fp_t (*h)[3]);
  void deviceToHostFp(kkFp3View d, fp_t (*h)[3]);
}
#endif
