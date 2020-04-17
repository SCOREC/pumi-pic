#pragma once
#include "ppTypes.h"
#include "ppMacros.h"
#include <type_traits>
namespace pumipic {

  template <class T, typename Space = Kokkos::DefaultExecutionSpace, typename ArrayLayout = Kokkos::LayoutLeft>
  class View {
  public:
    typedef Kokkos::View<T, ArrayLayout, Space> KView;
    typedef typename KView::execution_space execution_space;
    typedef typename KView::memory_space memory_space;
    typedef typename KView::device_type device_type;
    typedef typename KView::data_type data_type;
    typedef typename KView::value_type value_type;
    typedef View<T, typename KView::host_mirror_space, ArrayLayout> HostMirror;
    View() : view_() {}
    View(lid_t size) : view_("ppView", size) {}
    View(std::string name, lid_t size) : view_(name, size) {}
    View(const Kokkos::View<T, ArrayLayout, Space>& v) : view_(v) {}
    // View(const Kokkos::View<T, Space>& v) {Kokkos::deep_copy(view_,v);}
    // View(const Kokkos::View<T>& v) {Kokkos::deep_copy(view_,v);}

    static constexpr int rank = BaseType<T>::rank;

    //Direct access to the view/data pointer
    operator KView() const {return view_;}
    PP_INLINE KView* operator->() {return &view_;}
    PP_INLINE KView& view() {return view_;}
    PP_INLINE const T data() const {return view_.data();}

    PP_INLINE lid_t size() const {return view_.size();}
    PP_INLINE lid_t extent(int dim) const {return view_.extent(dim);}

    typedef typename BaseType<T>::type BT;
    // static_assert(BT::rank > 0, "ps Views of single values is not supported");
    // static_assert(BT::rank <= 4, "ps Views of rank greater than 4 is not supported");
    //Bracket operator for 1-dimentional arrays
    template <class U = T>
    PP_INLINE typename std::enable_if<BaseType<U>::rank == 1, BT>::type&
    operator[](const int& i) const {return view_[i];}
    //Parenthesis operator for 1-dimentional arrays
    template <class U = T>
    PP_INLINE typename std::enable_if<BaseType<U>::rank == 1, BT>::type&
    operator()(const int& i) const {return view_(i);}
    //Parenthesis operator for 2-dimentional arrays
    template <class U = T>
    PP_INLINE typename std::enable_if<BaseType<U>::rank == 2, BT>::type&
    operator()(const int& i, const int& j) const {return view_(i,j);}
    //Parenthesis operator for 3-dimentional arrays
    template <class U = T>
    PP_INLINE typename std::enable_if<BaseType<U>::rank == 3, BT>::type&
    operator()(const int& i, const int& j, const int& k) const {return view_(i,j,k);}
    //Parenthesis operator for 4-dimentional arrays
    template <class U = T>
    PP_INLINE typename std::enable_if<BaseType<U>::rank == 4, BT>::type&
    operator()(const int& i, const int& j, const int& k, const int& m) const {return view_(i,j,k,m);}

  private:
    KView view_;
  };

  template <class T, typename Space> struct CopyViewToView {
    PP_INLINE CopyViewToView(View<T*, Space> dst, int dst_index,
                             View<T*, Space> src, int src_index) {
      dst(dst_index) = src(src_index);
    }
  };
  template <class T, typename Space, int N> struct CopyViewToView<T[N], Space> {
    typedef T Type[N];
    PP_INLINE CopyViewToView(View<Type*, Space> dst, int dst_index,
                             View<Type*, Space> src, int src_index) {
      for (int i = 0; i < N; ++i)
        dst(dst_index, i) = src(src_index, i);
    }
  };
  template <class T, typename Space, int N, int M>
  struct CopyViewToView<T[N][M], Space> {
    typedef T Type[N][M];
    PP_INLINE CopyViewToView(View<Type*, Space> dst, int dst_index,
                             View<Type*, Space> src, int src_index) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
          src(src_index, i, j) = dst(dst_index, i, j);
    }
  };
  template <class T, typename Space, int N, int M, int P>
  struct CopyViewToView<T[N][M][P], Space> {
    typedef T Type[N][M][P];
    PP_INLINE CopyViewToView(View<Type*, Space> dst, int dst_index,
                             View<Type*, Space> src, int src_index) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
          for (int k = 0; k < P; ++k)
            dst(dst_index, i, j, k) = src(src_index, i, j, k);
    }
  };


}
