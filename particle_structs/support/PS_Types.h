#pragma once

#include <Kokkos_Core.hpp>

namespace pumipic {
  typedef int lid_t;
  typedef long int gid_t;

  typedef typename Kokkos::DefaultExecutionSpace::memory_space DefaultMemSpace;
}
