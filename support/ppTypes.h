#pragma once

#include <Kokkos_Core.hpp>

namespace pumipic {
  typedef int lid_t;
  typedef long int gid_t;

  typedef typename Kokkos::DefaultExecutionSpace::memory_space DefaultMemSpace;


  //Get the type with array lengths stripped off
  template <class T>
  struct BaseType {
    using type=T;
    static constexpr int size = 1;
    static constexpr int rank = 0;
  };
  template <class T, int N>
  struct BaseType<T[N]> {
    using type = typename BaseType<T>::type;
    static constexpr int size = N * BaseType<T>::size;
    static constexpr int rank = 1 + BaseType<T>::rank;
  };
  template <class T>
  struct BaseType<T*> {
    using type = typename BaseType<T>::type;
    static constexpr int size = 1 * BaseType<T>::size;
    static constexpr int rank = 1 + BaseType<T>::rank;
  };

}
