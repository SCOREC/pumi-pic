#pragma once

#include "MemberTypeLibraries.h"
#ifdef PP_ENABLE_CAB
  #include <Cabana_Core.hpp>
#endif
#include <type_traits>

namespace pumipic {
#ifndef PP_ENABLE_CAB
  //Dummy slice for when pumi-pic is build without Cabana
  //This is a hack that will go away in PSv2.0
  template <typename T>
  struct DummySlice {
    PP_INLINE T& access(int, int) const {return *x;}
    PP_INLINE T& access(int, int, int) const {return *x;}
    PP_INLINE T& access(int, int, int, int) const {return *x;}
    PP_INLINE T& access(int, int, int, int, int) const {return *x;}
    T* x;
  };
#endif

  //Forware declare subsegment
  template <typename Type, typename Device, typename MemoryAccessType=void,
            int VectorLength=1, int Stride=1>
  class SubSegment;


  template <typename Type, typename Device, typename MemoryAccessType=void,
            int VectorLength=1, int Stride=1>
  class Segment {
  public:
    using Base=typename BaseType<Type>::type;

    using ViewType = View<Type*, Device>;
#ifdef PP_ENABLE_CAB
    using SliceType = Cabana::Slice<Type, Device, MemoryAccessType,
                                    VectorLength, Stride>;
#else
    using SliceType = DummySlice<Base>;
#endif
    Segment() : is_view(false) {}
    Segment(ViewType v) : is_view(true), view(v) {}
    Segment(SliceType s) : is_view(false), slice(s) {}
    int getRank() { return std::rank<Type>::value; }
    template<int rankIdx> int getExtent() { return std::extent<Type,rankIdx>::value; }
    template <typename U, std::size_t N>
    using checkRank = typename std::enable_if<std::rank<Type>::value == N &&
                                              std::is_same<Type, U>::value,
                                              Base>::type;

    template <typename U = Type>
    PP_INLINE checkRank<U, 0>& operator()(const int& particle_index) const {
      if (is_view)
        return view(particle_index);
      else
        return slice.access(particle_index/VectorLength, particle_index%VectorLength);
    }
    template <typename U = Type>
    PP_INLINE checkRank<U, 1>& operator()(const int& particle_index,
                                          const int& i) const {
      if (is_view)
        return view(particle_index, i);
      else
        return slice.access(particle_index/VectorLength, particle_index%VectorLength,
                            i);
    }
    template <typename U = Type>
    PP_INLINE checkRank<U, 2>& operator()(const int& particle_index,
                                          const int& i, const int& j) const {
      if (is_view)
        return view(particle_index, i, j);
      else
        return slice.access(particle_index/VectorLength, particle_index%VectorLength,
                            i, j);
    }
    template <typename U = Type>
    PP_INLINE checkRank<U, 3>& operator()(const int& particle_index,
                                          const int& i, const int& j,
                                          const int& k) const {
      if (is_view)
        return view(particle_index, i, j, k);
      else
        return slice.access(particle_index/VectorLength, particle_index%VectorLength,
                            i, j, k);
    }


    PP_INLINE SubSegment<Type, Device, MemoryAccessType, VectorLength, Stride>
    getComponents(const int& particle_index) const {
      return SubSegment<Type, Device, MemoryAccessType,
                        VectorLength, Stride>(is_view,view, slice,particle_index);
    }

  private:
    bool is_view;
    ViewType view;
    SliceType slice;
  };


  template <typename Type, typename Device, typename MemoryAccessType,
            int VectorLength, int Stride>
  class SubSegment {
  public:
    using ViewType=View<Type*, Device>;
    using Base=typename BaseType<Type>::type;
#ifdef PP_ENABLE_CAB
    using SliceType = Cabana::Slice<Type, Device, MemoryAccessType,
                                    VectorLength, Stride>;
#else
    using SliceType = DummySlice<Base>;
#endif

    PP_INLINE SubSegment(bool is_v, const ViewType& view,
                         const SliceType& slice, const int& particle_index)
      : is_view(is_v), view_(view), slice_(slice), p(particle_index) {}
    PP_INLINE SubSegment(const SubSegment<Type, Device, MemoryAccessType,
                                          VectorLength, Stride>& old)
      : is_view(old.is_view), view_(old.view_), slice_(old.slice_),p(old.p) {}

    template <typename U, std::size_t N>
    using checkRank = typename std::enable_if<std::rank<Type>::value == N &&
                                              std::is_same<Type, U>::value,
                                              Base>::type;

    //Bracket operator for a 1D array
    template <class U = Type>
    PP_INLINE checkRank<U, 1>& operator[](const int& i) const {
      if (is_view)
        return view_(p, i);
      else
        return slice_.access(p / VectorLength, p % VectorLength, i);
    }

    //Parenthesis operator for single value
    template <class U = Type>
    PP_INLINE checkRank<U, 0>& operator()() const {
      if (is_view)
        return view_(p);
      else
        return slice_.access(p / VectorLength, p % VectorLength);
    }

    //Parenthesis operator for 1-dimentional arrays
    template <class U = Type>
    PP_INLINE checkRank<U, 1>& operator()(const int& i) const {
      if (is_view)
        return view_(p, i);
      else
        return slice_.access(p / VectorLength, p % VectorLength, i);
    }

    //Parenthesis operator for 2-dimentional arrays
    template <class U = Type>
    PP_INLINE checkRank<U, 2>& operator()(const int& i, const int& j) const {
      if (is_view)
        return view_(p, i,j);
      else
        return slice_.access(p / VectorLength, p % VectorLength, i, j);
    }

    //Parenthesis operator for 3-dimentional arrays
    template <class U = Type>
    PP_INLINE checkRank<U,3>& operator()(const int& i, const int& j,
                                         const int& k) const {
      if (is_view)
        return view_(p, i,j,k);
      else
        return slice_.access(p / VectorLength, p % VectorLength, i, j, k);
    }
  private:
    bool is_view;
    const ViewType& view_;
    const SliceType& slice_;
    const int p;

  };

}
