#pragma once

#include <Kokkos_Core.hpp>

namespace particle_structs {

template <class T>
  typename Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>::HostMirror deviceToHost(Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type> view) {
  typename Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>::HostMirror hv = 
    Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hv, view);
  return hv;
}
template <class T>
  void hostToDevice(Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>& view, T* data) {
  typename Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>::HostMirror hv = 
    Kokkos::create_mirror_view(view);
  for (size_t i = 0; i < hv.size(); ++i)
    hv(i) = data[i];
  Kokkos::deep_copy(view, hv);
}

template <typename T>
T getLastValue(Kokkos::View<T*> view) {
  const int size = view.size();
  if (size == 0)
    return 0;
  Kokkos::View<T*> lastValue("",1);
  Kokkos::parallel_for(1,KOKKOS_LAMBDA(const int& i) {
      lastValue(0) = view(size-1);
    });
  typename Kokkos::View<T*>::HostMirror host_view = deviceToHost<T>(lastValue);
  return host_view(0);
}

}
