#pragma once

#include "MemberTypeLibraries.h"
#include <ppArray.h>
#include <type_traits>

namespace pumipic {

template <typename Type, typename Device>
class Segment {
public:
  using Base=typename BaseType<Type>::type;

  using ViewType=View<Type*, Device>;
  Segment() {}
  Segment(ViewType v) : view(v){}

  template <typename U = Type>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<std::rank<Type>::value == 0 && std::is_same<U, Type>::value, Base>::type&
    operator()(const int& particle_index) const {
    return view(particle_index);
  }
  template <typename U = Type>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<std::rank<Type>::value == 1 && std::is_same<U, Type>::value, Base>::type&
    operator()(const int& particle_index, const int& i) const {
    return view(particle_index, i);
  }
  template <typename U = Type>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<std::rank<Type>::value == 2 && std::is_same<U, Type>::value, Base>::type&
    operator()(const int& particle_index, const int& i, const int& j) const {
    return view(particle_index, i, j);
  }
  template <typename U = Type>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<std::rank<Type>::value == 3 && std::is_same<U, Type>::value, Base>::type&
    operator()(const int& particle_index, const int& i, const int& j, const int& k) const {
    return view(particle_index, i, j, k);
  }


  template <int N, typename U = Type>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<std::rank<Type>::value == 1 && std::is_same<U, Type>::value, Array<Base, N> >::type
    getComponents(const int& particle_index) const {
    Array<Base, N> arr;
    for (int i = 0; i < N; ++i)
      arr[i] = view(particle_index, i);
    return arr;
  }
  template <int N, int M, typename U = Type>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<std::rank<Type>::value == 2 && std::is_same<U, Type>::value, Array<Base, N, M> >::type
    getComponents(const int& particle_index, const int& i, const int& j) const {
    Array<Base, N, M> arr;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        arr[i][j] = view(particle_index, i, j);
    return arr;
  }
  template <int N, int M, int K, typename U = Type>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<std::rank<Type>::value == 3 && std::is_same<U, Type>::value, Array<Base, N, M, K> >::type
    getComponents(const int& particle_index, const int& i, const int& j, const int& k) const {
    Array<Base, N, M> arr;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < K; ++k)
          arr[i][j][k] = view(particle_index, i, j, k);
    return arr;
  }

private:
  ViewType view;
};

}
