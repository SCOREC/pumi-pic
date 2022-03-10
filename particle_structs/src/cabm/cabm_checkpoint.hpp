#pragma once
#include <ppTiming.hpp>
#include <adios2.h>

namespace pumipic {
  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::checkpointWrite(std::string path) {
    const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("CabM checkpoint");
    Kokkos::Timer overall_timer;

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (!comm_rank) {
      fprintf(stderr, "doing adios2 stuff in cabm checkpoint\n");

      std::string config = "../../pumi-pic/pumipic-data/checkpoint/adios2.xml";
      adios2::ADIOS adios(config, MPI_COMM_WORLD);
      adios2::IO io = adios.DeclareIO("writerIO");

      adios2::Engine engine = io.Open(path, adios2::Mode::Write);

      // single-element variables saving
      
      adios2::Variable<lid_t> var_num_elems = io.DefineVariable<lid_t>("num_elems");
      adios2::Variable<lid_t> var_num_ptcls = io.DefineVariable<lid_t>("num_ptcls");
      adios2::Variable<lid_t> var_padding_start = io.DefineVariable<lid_t>("padding_start");
      adios2::Variable<double> var_extra_padding = io.DefineVariable<double>("extra_padding");
      adios2::Variable<std::string> var_name = io.DefineVariable<std::string>("name");

      // array-length variable saving
      adios2::Dims shape{static_cast<size_t>(num_elems)};
      adios2::Dims start{0};
      adios2::Dims count{static_cast<size_t>(num_elems)};
      //adios2::Variable<lid_t> var_ppe = io.DefineVariable<lid_t>("particles_per_element", shape, start, count);
      adios2::Variable<gid_t> var_gids = io.DefineVariable<gid_t>("element_gids", shape, start, count);
      //shape = adios2::Dims(num_ptcls);
      //start = adios2::Dims(0);
      //count = adios2::Dims(num_ptcls);
      //adios2::Variable<lid_t> var_ptcl_elems = io.DefineVariable<lid_t>("particle_elements", shape, start, count);

      // TODO setup for var_ppe and var_ptcl_elems
      //kkLidHostMirror offsets_h = deviceToHost(offsets);
      kkGidHostMirror element_to_gid_h = deviceToHost(element_to_gid);

      // TODO: AoSoA variable saving

      engine.BeginStep();
      engine.Put(var_num_elems, num_elems);
      engine.Put(var_num_ptcls, num_ptcls);
      engine.Put(var_padding_start, padding_start);
      engine.Put(var_extra_padding, extra_padding);
      //engine.Put(var_name, name);

      engine.Put(var_gids, element_to_gid_h.data());
      engine.EndStep();

      engine.Close();
    }

    RecordTime("CabM checkpoint", overall_timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::checkpointRead(std::string path) {
    const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("CabM checkpointRevert");
    Kokkos::Timer overall_timer;

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (!comm_rank) {
      fprintf(stderr, "doing adios2 stuff in cabm checkpointRevert\n");

      std::string config = "../../pumi-pic/pumipic-data/checkpoint/adios2.xml";
      adios2::ADIOS adios(config, MPI_COMM_WORLD);
      adios2::IO io = adios.DeclareIO("readerIO");

      adios2::Engine engine = io.Open(path, adios2::Mode::Read);

      // single-element variables
      adios2::Variable<lid_t> var_num_elems = io.InquireVariable<lid_t>("num_elems");
      adios2::Variable<lid_t> var_num_ptcls = io.InquireVariable<lid_t>("num_ptcls");
      adios2::Variable<lid_t> var_padding_start = io.InquireVariable<lid_t>("padding_start");
      adios2::Variable<double> var_extra_padding = io.InquireVariable<double>("extra_padding");
      //adios2::Variable<std::string> var_name = io.InquireVariable<std::string>("name");

      engine.BeginStep();
      engine.Get(var_num_elems, num_elems);
      engine.Get(var_num_ptcls, num_ptcls);
      engine.Get(var_padding_start, padding_start);
      engine.Get(var_extra_padding, extra_padding);
      //engine.Get(var_name, name);
      engine.EndStep();

      // array-sized variables
      adios2::Variable<gid_t> var_gids = io.InquireVariable<gid_t>("element_gids");
      Kokkos::View<gid_t*,host_space> element_gids_h(Kokkos::ViewAllocateWithoutInitializing("element_gids_h"), num_elems);
      
      // TODO: AoSoA variable

      // TODO: Check if types are the same before moving on

      engine.BeginStep();
      engine.Get(var_gids, element_gids_h.data());
      engine.EndStep();

      kkGidView element_gids(Kokkos::ViewAllocateWithoutInitializing("element_gids"), num_elems);
      hostToDevice(element_gids, element_gids_h.data());

      engine.Close();
    }

    RecordTime("CabM checkpointRevert", overall_timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

}
