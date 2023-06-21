#pragma once

#include <ppMacros.h>
namespace pumipic {
struct MyPair {
  KOKKOS_FORCEINLINE_FUNCTION constexpr MyPair() : first(0), second(0) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr MyPair(int i) : first(i), second(0) {}
  KOKKOS_INLINE_FUNCTION MyPair& operator=(const MyPair& p) {
    if (this != &p) {
      first = p.first;
      second = p.second;
    }
    return *this;
  }
  KOKKOS_FORCEINLINE_FUNCTION void operator=(const volatile MyPair& p) volatile {
    first = p.first;
    second = p.second;
  }
  KOKKOS_FORCEINLINE_FUNCTION int operator-(const volatile MyPair& p) const volatile { return first - p.first;}
  KOKKOS_FORCEINLINE_FUNCTION bool operator==(const volatile MyPair& p) const volatile {return first==p.first;}
  KOKKOS_FORCEINLINE_FUNCTION bool operator!=(const volatile MyPair& p) const volatile {return !(*this == p);}
  //Reverse operators in order to get largest first
  KOKKOS_FORCEINLINE_FUNCTION bool operator<(const volatile MyPair& p) const volatile {return first > p.first || (first ==p.first && second < p.second);}
  KOKKOS_FORCEINLINE_FUNCTION bool operator>(const volatile MyPair& p) const volatile {return first < p.first || (first==p.first && second > p.second);}
  int first, second;
};

}

#ifdef PP_USE_GPU
namespace Kokkos {
  using pumipic::MyPair;
  PP_DEVICE_VAR MyPair ma = MyPair(10000000);
  PP_DEVICE_VAR MyPair mi = MyPair(0);
  template <>
  struct reduction_identity<MyPair> {
    KOKKOS_FORCEINLINE_FUNCTION constexpr static const MyPair& max() {return ma;}
    KOKKOS_FORCEINLINE_FUNCTION constexpr static const  MyPair& min() {return mi;}
  };
}
#endif
