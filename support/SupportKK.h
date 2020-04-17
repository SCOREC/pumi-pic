#pragma once

#include <Kokkos_Core.hpp>
#include "ppView.h"
#include "ppMacros.h"
#ifdef PP_USE_CUDA
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#endif
namespace pumipic {

  /* template <typename ExecSpace> struct ThrustSpace; */
  /* template <> struct ThrustSpace<Kokkos::HostSpace> { */
  /*   static constexpr thrust::detail::host_t space=thrust::host; */
  /* }; */
  /* template <> struct ThrustSpace<Kokkos::CudaSpace> { */
  /*   static constexpr thrust::detail::device_t space=thrust::device; */
  /* }; */
  /* Wrapper around performing inclusive/exclusive scans
     Note: These exist because kokkos scans on Summit/Aimos fail on occassion,
             however thrust scans work

     The wrapper works for both Kokkos views and pumipic views on the device only
   */
  template <typename ViewT>
  void exclusive_scan(ViewT entries, ViewT result) {
#ifdef PP_USE_CUDA
    thrust::exclusive_scan(thrust::device /*ThrustSpace<ViewT::memory_space>::space */,
                           entries.data(), entries.data() + entries.size(), result.data(), 0);
#else
    auto exclusive_sum = KOKKOS_LAMBDA(const int index, typename ViewT::value_type& cur, const bool final) {
      if (final) {
        result(index) = cur;
      }
      cur += entries(index);
    };
    Kokkos::parallel_scan("exclusive_scan", entries.size(), exclusive_sum);
#endif
  }
  template <typename ViewT>
  void inclusive_scan(ViewT entries, ViewT result) {
#ifdef PP_USE_CUDA
    thrust::inclusive_scan(thrust::device /*ThrustSpace<ViewT::memory_space>::space */,
                           entries.data(), entries.data() + entries.size(), result.data(), 0);
#else
    auto inclusive_sum = KOKKOS_LAMBDA(const int index, typename ViewT::value_type& cur, const bool final) {
      cur += entries(index);
      if (final) {
        result(index) = cur;
      }
    };
    Kokkos::parallel_scan("inclusive_scan", entries.size(), inclusive_sum);
#endif
  }
  /* Taken from https://stackoverflow.com/questions/31762958/check-if-class-is-a-template-specialization
     Checks if type is a specialization of a class template
   */
  template <class T, template <class...> class Template>
  struct is_specialization : std::false_type {};

  template <template <class...> class Template, class... Args>
  struct is_specialization<Template<Args...>, Template> : std::true_type {};

  //Checks if view is a kokkos view and provides the type T
  template <typename ViewT, typename T = ViewT>
  using IsKokkosView =
    typename std::enable_if<is_specialization<ViewT, Kokkos::View>{}, T>::type;
  //Checks if view is a pumipic view and provides the type T
  template <typename ViewT, typename T = ViewT>
  using IsPPView =
    typename std::enable_if<is_specialization<ViewT, View>{}, T>::type;

  template <class ViewT> typename
  IsKokkosView<ViewT>::HostMirror create_mirror_view(ViewT v) {
    return Kokkos::create_mirror_view(v);
  }
  template <class ViewT> typename
  IsPPView<ViewT>::HostMirror create_mirror_view(ViewT v) {
    return typename ViewT::HostMirror(Kokkos::create_mirror_view(v.view()));
  }

  template <class ViewT, class ViewT2>
  IsKokkosView<ViewT, void> deep_copy(ViewT dst, ViewT2 src) {
    Kokkos::deep_copy(dst, src);
  }
  template <class ViewT, class ViewT2>
  IsPPView<ViewT, void> deep_copy(ViewT dst, ViewT2 src) {
    Kokkos::deep_copy(dst.view(), src.view());
  }

  template <class ViewT>
  typename ViewT::HostMirror deviceToHost(ViewT view) {
    auto hv = create_mirror_view(view);
    deep_copy(hv, view);
    return hv;
  }

  template <class ViewT, class T>
  typename std::enable_if<ViewT::rank==1>::type
  hostToDevice(typename ViewT::HostMirror hv, ViewT, T* data) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      hv(i) = data[i];
  }
  template <class ViewT, class T>
  typename std::enable_if<ViewT::rank==2>::type
  hostToDevice(typename ViewT::HostMirror hv, ViewT, T* data) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        hv(i,j) = data[i][j];
  }
  template <class ViewT, class T>
  typename std::enable_if<ViewT::rank==3>::type
  hostToDevice(typename ViewT::HostMirror hv, ViewT, T* data) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        for (size_t k = 0; k < hv.extent(2); ++k)
          hv(i,j,k) = data[i][j][k];
  }
  template <class ViewT, class T>
  typename std::enable_if<ViewT::rank==4>::type
  hostToDevice(typename ViewT::HostMirror hv, ViewT, T* data) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        for (size_t k = 0; k < hv.extent(2); ++k)
          for (size_t l = 0; l < hv.extent(3); ++l)
            hv(i,j,k,l) = data[i][j][k][l];
  }

  template <class ViewT, class T>
  void hostToDevice(ViewT view, T* data) {
    auto hv = create_mirror_view(view);
    hostToDevice(hv,view,data);
    deep_copy(view, hv);
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

  template <typename ViewT>
  PP_INLINE typename std::enable_if<ViewT::rank == 1>::type copyViewToView(ViewT dst, int dstind,
                                                                           ViewT src, int srcind){
    dst(dstind) = src(srcind);
  }
  template <typename ViewT>
  PP_INLINE typename std::enable_if<ViewT::rank == 2>::type copyViewToView(ViewT dst, int dstind,
                                                                           ViewT src, int srcind){
    for (int i = 0; i < dst.extent(1); ++i)
      dst(dstind, i) = src(srcind, i);
  }
  template <typename ViewT>
  PP_INLINE typename std::enable_if<ViewT::rank == 3>::type copyViewToView(ViewT dst, int dstind,
                                                                           ViewT src, int srcind){
    for (int i = 0; i < dst.extent(1); ++i)
      for (int j = 0; j < dst.extent(2); ++j)
        dst(dstind, i, j) = src(srcind, i, j);
  }
  template <typename ViewT>
  PP_INLINE typename std::enable_if<ViewT::rank == 4>::type copyViewToView(ViewT dst, int dstind,
                                                                           ViewT src, int srcind){
    for (int i = 0; i < dst.extent(1); ++i)
      for (int j = 0; j < dst.extent(2); ++j)
        for (int k = 0; k < dst.extent(3); ++k)
          dst(dstind, i, j, k) = src(srcind, i, j, k);
  }

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
