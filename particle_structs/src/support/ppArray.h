#pragma once

namespace pumipic {

  template <typename T, std::size_t... Dims>
  class Array;

  template <typename T, std::size_t Dim>
  class Array<T, Dim> {
    T array_[Dim];
  public:
    using value_type = T;

    PP_INLINE T& operator[](int i) {return array_[i];}
    PP_INLINE T const& operator[](int i) const {return array_[i];}
    PP_INLINE T* data() { return array_; }
    PP_INLINE T const* data() const { return array_; }

    PP_INLINE Array() {}
    PP_INLINE Array(Array<T, Dim> const& rhs) {
      for (int i = 0; i < Dim; ++i) array_[i] = rhs[i];
    }
    PP_INLINE void operator=(Array<T, Dim> const& rhs) {
      for (int i = 0; i < Dim; ++i) array_[i] = rhs[i];
    }

    PP_INLINE std::size_t size() {return Dim;}
  };

  template <typename T, std::size_t Dim, std::size_t... Dims>
  class Array<T, Dim, Dims...> {
    Array<T, Dims...> array_[Dim];
  public:
    using value_type = T;

    PP_INLINE Array<T, Dims...>& operator[](int i) {return array_[i];}
    PP_INLINE Array<T, Dims...> const& operator[](int i) const {return array_[i];}

    PP_INLINE Array() {}
    PP_INLINE Array(Array<T, Dim, Dims...> const& rhs) {
      for (int i = 0; i < Dim; ++i) array_[i] = rhs[i];
    }
    PP_INLINE void operator=(Array<T, Dim, Dims...> const& rhs) {
      for (int i = 0; i < Dim; ++i) array_[i] = rhs[i];
    }

    PP_INLINE std::size_t size() {return Dim;}
  };

}
