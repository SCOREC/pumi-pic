#pragma once

#include "MemberTypeLibraries.h"
#include <type_traits>
namespace particle_structs {

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

private:
  ViewType view;
};

}
