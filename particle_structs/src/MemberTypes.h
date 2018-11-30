#ifndef __MEMBERTYPES_H__
#define __MEMBERTYPES_H__

#include <cstdlib>
template<std::size_t N, typename T, typename... Types>
struct MemberSize;

template<typename T, typename... Types>
struct MemberSize<0, T,Types...> {
  static constexpr std::size_t memsize = 0;
};

template<std::size_t N, typename T, typename... Types>
struct MemberSize {
  static constexpr std::size_t memsize = sizeof(T) + MemberSize<N-1, Types...>::memsize;
};

template<typename... Types>
struct MemberTypes;

template<>
struct MemberTypes<> {
  static constexpr std::size_t size = 0;
  static constexpr std::size_t memsize = 0;
};

template<typename H, typename... T>
  struct MemberTypes<H,T...> {
  static constexpr std::size_t size = 1 + MemberTypes<T...>::size;
  static constexpr std::size_t memsize = sizeof(H) + MemberTypes<T...>::memsize;

  template <std::size_t I>
    static std::size_t sizeToIndex() {return MemberSize<I,H,T...,void>::memsize;}
};

template<std::size_t N, typename... Types>
struct MemberTypeAtIndexImpl;


template<typename T, typename... Types>
struct MemberTypeAtIndexImpl<0, T,Types...> {
  using type = T;
};

template<std::size_t N, typename T, typename... Types>
  struct MemberTypeAtIndexImpl<N, T, Types...> {
  using type = typename MemberTypeAtIndexImpl<N-1 , Types...>::type;
};


template<std::size_t N, typename... Types>
struct MemberTypeAtIndex;

template<std::size_t N, typename... Types>
struct MemberTypeAtIndex<N,MemberTypes<Types...> > {
  using type = typename MemberTypeAtIndexImpl<N, Types...>::type;
};


#endif
