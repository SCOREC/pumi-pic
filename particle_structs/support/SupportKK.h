#pragma once

#include <Kokkos_Core.hpp>

namespace particle_structs {

  template <class View>
  typename View::HostMirror deviceToHost(View view) {
    typename View::HostMirror hv = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hv, view);
  return hv;
}
template <class T, typename ExecSpace>
  void hostToDevice(Kokkos::View<T*, ExecSpace> view, T* data) {
  typename Kokkos::View<T*, ExecSpace>::HostMirror hv = Kokkos::create_mirror_view(view);
  for (size_t i = 0; i < hv.size(); ++i)
    hv(i) = data[i];
  Kokkos::deep_copy(view, hv);
}

template <typename T, typename ExecSpace>
T getLastValue(Kokkos::View<T*, ExecSpace> view) {
  const int size = view.size();
  if (size == 0)
    return 0;
  Kokkos::View<T*, ExecSpace> lastValue("",1);
  Kokkos::parallel_for(1,KOKKOS_LAMBDA(const int& i) {
      lastValue(0) = view(size-1);
    });
  typename Kokkos::View<T*, ExecSpace>::HostMirror host_view = deviceToHost(lastValue);
  return host_view(0);
}

template <class T, typename ExecSpace> struct CopyViewToView {
  KOKKOS_INLINE_FUNCTION CopyViewToView(Kokkos::View<T*, ExecSpace> dst, int dst_index,
                                              Kokkos::View<T*, ExecSpace> src, int src_index) {
    dst(dst_index) = src(src_index);
  }
};
template <class T, typename ExecSpace, int N> struct CopyViewToView<T[N], ExecSpace> {
  KOKKOS_INLINE_FUNCTION CopyViewToView(Kokkos::View<T*[N], ExecSpace> dst, int dst_index,
                                              Kokkos::View<T*[N], ExecSpace> src, int src_index) {
    for (int i = 0; i < N; ++i)
      dst(dst_index, i) = src(src_index, i);
  }
};
template <class T, typename ExecSpace, int N, int M> 
struct CopyViewToView<T[N][M], ExecSpace> {
  KOKKOS_INLINE_FUNCTION CopyViewToView(Kokkos::View<T*[N][M], ExecSpace> dst, int dst_index,
                                              Kokkos::View<T*[N][M], ExecSpace> src, int src_index) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        src(src_index, i, j) = dst(dst_index, i, j);
  }
};
template <class T, typename ExecSpace, int N, int M, int P> 
  struct CopyViewToView<T[N][M][P], ExecSpace> {
  KOKKOS_INLINE_FUNCTION CopyViewToView(Kokkos::View<T*[N][M][P], ExecSpace> dst, 
                                              int dst_index,
                                              Kokkos::View<T*[N][M][P], ExecSpace> src, 
                                              int src_index) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < P; ++k)
          dst(dst_index, i, j, k) = src(src_index, i, j, k);
  }
};

  template <typename T, typename ExecSpace> struct Subview {
    static auto subview(Kokkos::View<T*, ExecSpace> view, const std::pair<int,int>& range)
      -> decltype(Kokkos::subview(view, range)) {
      return Kokkos::subview(view, range);
    }
  };
  template <typename T, typename ExecSpace, size_t N> struct Subview<T[N],ExecSpace> {
    static auto subview(Kokkos::View<T*[N], ExecSpace> view, const std::pair<int,int>& range)
      -> decltype(Kokkos::subview(view, range, Kokkos::ALL())) {
      return Kokkos::subview(view, range, Kokkos::ALL());
    }
  };
  template <typename T, typename ExecSpace, size_t N, size_t M>
  struct Subview<T[N][M],ExecSpace> {
    static auto subview(Kokkos::View<T*[N][M], ExecSpace> view,
                           const std::pair<int,int>& range)
      -> decltype(Kokkos::subview(view, range, Kokkos::ALL(), Kokkos::ALL())) {
      return Kokkos::subview(view, range, Kokkos::ALL(), Kokkos::ALL());
    }
  };
  template <typename T, typename ExecSpace, size_t N, size_t M, size_t P>
  struct Subview<T[N][M][P],ExecSpace> {
    static auto subview(Kokkos::View<T*[N][M][P], ExecSpace> view,
                           const std::pair<int,int>& range)
      -> decltype(Kokkos::subview(view, range, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL())) {
      return Kokkos::subview(view, range, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    }
  };



}
