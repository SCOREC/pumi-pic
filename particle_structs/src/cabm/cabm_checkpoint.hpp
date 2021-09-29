#pragma once
#include <ppTiming.hpp>
#include <adios2.h>

namespace pumipic {
  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::checkpoint(std::string path) {
    const auto btime = prebarrier();

    Kokkos::Profiling::pushRegion("CabM checkpoint");
    Kokkos::Timer overall_timer;

    fprintf(stderr, "doing adios2 stuff in cabm checkpoint\n");
    adios2::ADIOS adios(MPI_COMM_WORLD); //hack, need to pass this in
    adios2::IO fooIO = adios.DeclareIO("Foo");

    RecordTime("CabM checkpoint", overall_timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

}
