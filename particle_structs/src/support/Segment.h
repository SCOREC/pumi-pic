#pragma once

#include "MemberTypeLibraries.h"
#include <type_traits>

namespace pumipic {

  //Forware declare subsegment
  template <typename Type, typename Device>
  class SubSegment;


  template <typename Type, typename Device>
  class Segment {
  public:
    using Base=typename BaseType<Type>::type;

    using ViewType=View<Type*, Device>;
    Segment() {}
    Segment(ViewType v) : view(v){}

    template <typename U, std::size_t N>
    using checkRank = typename std::enable_if<std::rank<Type>::value == N &&
                                              std::is_same<Type, U>::value,
                                              Base>::type;

    template <typename U = Type>
    PP_INLINE checkRank<U, 0>& operator()(const int& particle_index) const {
      return view(particle_index);
    }
    template <typename U = Type>
    PP_INLINE checkRank<U, 1>& operator()(const int& particle_index,
                                          const int& i) const {
      return view(particle_index, i);
    }
    template <typename U = Type>
    PP_INLINE checkRank<U, 2>& operator()(const int& particle_index,
                                          const int& i, const int& j) const {
      return view(particle_index, i, j);
    }
    template <typename U = Type>
    PP_INLINE checkRank<U, 3>& operator()(const int& particle_index,
                                          const int& i, const int& j,
                                          const int& k) const {
      return view(particle_index, i, j, k);
    }


    PP_INLINE SubSegment<Type, Device> getComponents(const int& particle_index) const {
      return SubSegment<Type, Device>(view, particle_index);
    }

  private:
    ViewType view;
  };


  template <typename Type, typename Device>
  class SubSegment {
  public:
    using ViewType=View<Type*, Device>;
    using Base=typename BaseType<Type>::type;

    PP_INLINE SubSegment(const ViewType& view, const int& particle_index)
      : view_(view), p(particle_index) {}
    PP_INLINE SubSegment(const SubSegment<Type, Device>& old)
      : view_(old.view_), p(old.p) {}

    template <typename U, std::size_t N>
    using checkRank = typename std::enable_if<std::rank<Type>::value == N &&
                                              std::is_same<Type, U>::value,
                                              Base>::type;

    //Bracket operator for a 1D array
    template <class U = Type>
    PP_INLINE checkRank<U, 1>& operator[](const int& i) const {
      return view_(p, i);
    }

    //Parenthesis operator for single value
    template <class U = Type>
    PP_INLINE checkRank<U, 0>& operator()() const {return view_(p);}

    //Parenthesis operator for 1-dimentional arrays
    template <class U = Type>
    PP_INLINE checkRank<U, 1>& operator()(const int& i) const {
      return view_(p, i);
    }

    //Parenthesis operator for 2-dimentional arrays
    template <class U = Type>
    PP_INLINE checkRank<U, 2>& operator()(const int& i, const int& j) const {
      return view_(p, i,j);
    }

    //Parenthesis operator for 3-dimentional arrays
    template <class U = Type>
    PP_INLINE checkRank<U,3>& operator()(const int& i, const int& j,
                                         const int& k) const {
      return view_(p, i,j,k);
    }
  private:
    const ViewType& view_;
    const int p;

  };

}
