#pragma once

#include <Kokkos_Core.hpp>

namespace particle_structs {

  template <class View>
    typename View::HostMirror deviceToHost(View view) {
    auto hv = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hv, view);
    return hv;
  }

 template <class T, typename Device> struct HostToDevice;

 template <class T, typename Device>
 void hostToDevice(Kokkos::View<T*, Device> view, T* data) {
   HostToDevice<T, Device>(view, data);
 }

template <typename T, typename Device>
T getLastValue(Kokkos::View<T*, Device> view) {
  const int size = view.size();
  if (size == 0)
    return 0;
  T lastVal;
  Kokkos::deep_copy(lastVal,Kokkos::subview(view,size-1));
  return lastVal;
}

template <class T, typename Device> struct CopyViewToView {
  KOKKOS_INLINE_FUNCTION CopyViewToView(Kokkos::View<T*, Device> dst, int dst_index,
                                              Kokkos::View<T*, Device> src, int src_index) {
    dst(dst_index) = src(src_index);
  }
};
template <class T, typename Device, int N> struct CopyViewToView<T[N], Device> {
  KOKKOS_INLINE_FUNCTION CopyViewToView(Kokkos::View<T*[N], Device> dst, int dst_index,
                                              Kokkos::View<T*[N], Device> src, int src_index) {
    for (int i = 0; i < N; ++i)
      dst(dst_index, i) = src(src_index, i);
  }
};
template <class T, typename Device, int N, int M>
struct CopyViewToView<T[N][M], Device> {
  KOKKOS_INLINE_FUNCTION CopyViewToView(Kokkos::View<T*[N][M], Device> dst, int dst_index,
                                              Kokkos::View<T*[N][M], Device> src, int src_index) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        src(src_index, i, j) = dst(dst_index, i, j);
  }
};
template <class T, typename Device, int N, int M, int P>
  struct CopyViewToView<T[N][M][P], Device> {
  KOKKOS_INLINE_FUNCTION CopyViewToView(Kokkos::View<T*[N][M][P], Device> dst,
                                              int dst_index,
                                              Kokkos::View<T*[N][M][P], Device> src,
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

  template <class T, typename Device> struct HostToDevice {
    HostToDevice(Kokkos::View<T*, Device> view, T* data) {
      typename Kokkos::View<T*, Device>::HostMirror hv = Kokkos::create_mirror_view(view);
      for (size_t i = 0; i < hv.size(); ++i)
        hv(i) = data[i];
      Kokkos::deep_copy(view, hv);
    }
  };
  template <class T, typename Device, std::size_t N> struct HostToDevice<T[N], Device> {
    HostToDevice(Kokkos::View<T*[N], Device> view, T (*data)[N]) {
      typename Kokkos::View<T*[N], Device>::HostMirror hv = Kokkos::create_mirror_view(view);
      for (size_t i = 0; i < hv.extent(0); ++i)
        for (size_t j = 0; j < N; ++j)
          hv(i,j) = data[i][j];
      Kokkos::deep_copy(view, hv);
    }
  };


}
