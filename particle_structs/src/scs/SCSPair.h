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

