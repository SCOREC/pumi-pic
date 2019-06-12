#pragma once

#include <Kokkos_Core.hpp>

namespace particle_structs {

template <class View>
typename View::HostMirror deviceToHost(View view) {
  auto hv = Kokkos::create_mirror_view(view);
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

  template <typename T> struct Subview {
    template <typename View>
    static View subview(View view, int start, int size) {
      View new_view("subview", size);
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int& i) {
        new_view(i) = view(start + i);
      });
      return new_view;
    }
  };
  template <typename T, size_t N> struct Subview<T[N]> {
    template <typename View>
    static View subview(View view, int start, int size) {
      View new_view("subview", size);
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int& i) {
        for (int j = 0; j < N; ++j)
          new_view(i,j) = view(start + i,j);
      });
      return new_view;
    }
  };
  template <typename T, size_t N, size_t M>
  struct Subview<T[N][M]> {
    template <typename View>
    static View subview(View view, int start, int size) {
      View new_view("subview", size);
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int& i) {
        for (int j = 0; j < N; ++j)
          for (int k = 0; k < M; ++k)
            new_view(i,j,k) = view(start + i,j,k);
      });
      return new_view;
    }
  };
  template <typename T, size_t N, size_t M, size_t P>
  struct Subview<T[N][M][P]> {
    template <typename View>
    static View subview(View view, int start, int size) {
      View new_view("subview", size);
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int& i) {
        for (int j = 0; j < N; ++j)
          for (int k = 0; k < M; ++k)
            for (int l = 0; l < P; ++l)
              new_view(i,j,k,l) = view(start + i,j,k,l);
      });
      return new_view;

    }
  };



}
