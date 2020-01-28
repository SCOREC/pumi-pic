#pragma once
#include "PS_Types.h"
#include "PS_Macros.h"
#include "MemberTypes.h"
#include <type_traits>
namespace particle_structs {

  template <class T, typename Space = Kokkos::DefaultExecutionSpace, typename ArrayLayout = Kokkos::LayoutLeft>
  class View {
  public:
    typedef Kokkos::View<T, ArrayLayout, Space> KView;
    typedef View<T, typename KView::host_mirror_space, ArrayLayout> HostMirror;
    View() : view_() {}
    View(lid_t size) : view_("psView", size) {}
    View(std::string name, lid_t size) : view_(name, size) {}
    View(const Kokkos::View<T, ArrayLayout, Space>& v) : view_(v) {}
    // View(const Kokkos::View<T, Space>& v) {Kokkos::deep_copy(view_,v);}
    // View(const Kokkos::View<T>& v) {Kokkos::deep_copy(view_,v);}

    static constexpr int rank = BaseType<T>::rank;

    //Direct access to the view/data pointer
    operator KView() const {return view_;}
    PS_INLINE KView* operator->() {return &view_;}
    PS_INLINE KView& view() {return view_;}
    PS_INLINE const T& data() const {return view_.data();}

    PS_INLINE lid_t size() const {return view_.size();}
    PS_INLINE lid_t extent(int dim) const {return view_.extent(dim);}

    typedef typename BaseType<T>::type BT;
    // static_assert(BT::rank > 0, "ps Views of single values is not supported");
    // static_assert(BT::rank <= 4, "ps Views of rank greater than 4 is not supported");
    //Bracket operator for 1-dimentional arrays
    template <class U = T>
    PS_INLINE typename std::enable_if<BaseType<U>::rank == 1, BT>::type&
    operator[](const int& i) const {return view_[i];}
    //Parenthesis operator for 1-dimentional arrays
    template <class U = T>
    PS_INLINE typename std::enable_if<BaseType<U>::rank == 1, BT>::type&
    operator()(const int& i) const {return view_(i);}
    //Parenthesis operator for 2-dimentional arrays
    template <class U = T>
    PS_INLINE typename std::enable_if<BaseType<U>::rank == 2, BT>::type&
    operator()(const int& i, const int& j) const {return view_(i,j);}
    //Parenthesis operator for 3-dimentional arrays
    template <class U = T>
    PS_INLINE typename std::enable_if<BaseType<U>::rank == 3, BT>::type&
    operator()(const int& i, const int& j, const int& k) const {return view_(i,j,k);}
    //Parenthesis operator for 4-dimentional arrays
    template <class U = T>
    PS_INLINE typename std::enable_if<BaseType<U>::rank == 4, BT>::type&
    operator()(const int& i, const int& j, const int& k, const int& m) const {return view_(i,j,k,m);}

  private:
    KView view_;
  };

  template <class ViewT> typename ViewT::HostMirror create_mirror_view(ViewT v) {
    return typename ViewT::HostMirror(Kokkos::create_mirror_view(v.view()));
  }

  template <class ViewT, class ViewT2>
  void deep_copy(ViewT dst, ViewT2 src) {
    Kokkos::deep_copy(dst.view(), src.view());
  }

  template <class T, typename Space> struct CopyViewToView {
    PS_INLINE CopyViewToView(View<T*, Space> dst, int dst_index,
                             View<T*, Space> src, int src_index) {
      dst(dst_index) = src(src_index);
    }
  };
  template <class T, typename Space, int N> struct CopyViewToView<T[N], Space> {
    typedef T Type[N];
    PS_INLINE CopyViewToView(View<Type*, Space> dst, int dst_index,
                             View<Type*, Space> src, int src_index) {
      for (int i = 0; i < N; ++i)
        dst(dst_index, i) = src(src_index, i);
    }
  };
  template <class T, typename Space, int N, int M>
  struct CopyViewToView<T[N][M], Space> {
    typedef T Type[N][M];
    PS_INLINE CopyViewToView(View<Type*, Space> dst, int dst_index,
                             View<Type*, Space> src, int src_index) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
          src(src_index, i, j) = dst(dst_index, i, j);
    }
  };
  template <class T, typename Space, int N, int M, int P>
  struct CopyViewToView<T[N][M][P], Space> {
    typedef T Type[N][M][P];
    PS_INLINE CopyViewToView(View<Type*, Space> dst, int dst_index,
                             View<Type*, Space> src, int src_index) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
          for (int k = 0; k < P; ++k)
            dst(dst_index, i, j, k) = src(src_index, i, j, k);
    }
  };


}
