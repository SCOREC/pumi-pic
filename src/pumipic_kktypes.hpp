#ifndef PUMIPIC_KKTYPES_H
#define PUMIPIC_KKTYPES_H

#include <Kokkos_Core.hpp>
#include <particle_structs.hpp>
#include <chrono>
#include <thread>
#include <ppTypes.h>

namespace pumipic {
#ifdef FP64
  typedef double fp_t;
#endif
#ifdef FP32
  typedef float fp_t;
#endif
  typedef fp_t Vector3d[3];

  typedef Kokkos::DefaultExecutionSpace exe_space;
  typedef exe_space::device_type device_type;

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
