#ifndef PUMIPIC_PROFILING_H
#define PUMIPIC_PROFILING_H

#include "Kokkos_Core.hpp"
#include <mpi.h>

void pumipic_enable_prebarrier();

double pumipic_prebarrier(MPI_Comm mpi_comm);

#endif
