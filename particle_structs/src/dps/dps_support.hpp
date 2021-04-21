#pragma once
#include <Cabana_Core.hpp>
#include "cabana_support.hpp"

namespace pumipic {
  /* Class which appends a type T to a pp::MemberType and provides it as a
       cabana::MemberType
     Usage: typename AppendMT<Type, MemberTypes>::type
  */
  template <typename T, typename... Types> struct AppendMT;

}