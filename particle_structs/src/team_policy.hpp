
Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> TeamPolicyAuto(int league_size, int team_size)
{
    return Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(league_size, Kokkos::AUTO());
}