#pragma once

#include <Omega_h_library.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace pumipic {
  class Library {
  public:
    Library(int* argc, char*** argv);
    ~Library();
    Omega_h::Library& omega_h_lib() {return *oh_lib;}
  private:
    Omega_h::Library* oh_lib;
    bool own_kokkos;
    bool own_pcu;
    bool own_mpi;
  };
}
