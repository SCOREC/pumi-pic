#pragma once

namespace pumipic{
    Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> TeamPolicyAuto(int league_size, int team_size)
    {
    #ifdef PP_USE_CUDA
        return Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(league_size, team_size);
    #else
        return Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(league_size, Kokkos::AUTO());
    #endif
    }
}