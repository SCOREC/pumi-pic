#pragma once
#include <Cabana_Core.hpp>

namespace pumipic {
  /* Class which appends a type T to a pp::MemberType and provides it as a
       cabana::MemberType
     Usage: typename AppendMT<Type, MemberTypes>::type
  */
  template <typename T, typename... Types> struct AppendMT;

  //Append type to the end
  template <typename T, typename... Types>
  struct AppendMT<T, particle_structs::MemberTypes<Types...> > {
    static constexpr int size = 1 + Cabana::MemberTypes<Types...>::size;
    using type = Cabana::MemberTypes<Types..., T>; //Put T before Types... to put at beginning
  };

  template <typename DataTypes> using PS_DTBool = typename AppendMT<bool,DataTypes>::type;
}