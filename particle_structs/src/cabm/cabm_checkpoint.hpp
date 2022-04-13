#pragma once
#include <ppTiming.hpp>
#include <adios2.h>
#include <type_traits>

namespace pumipic {

  // Templated AoSoA Write
  template <typename PS, typename... Types> struct AoSoAPut;
  template <typename Device, std::size_t M, typename CMDT, typename ViewT, typename... Types> struct AoSoAPutImpl;
  //Per type Adios2::Put for AoSoA
  template <typename PS, std::size_t M, typename CMDT, typename ViewT>
  struct AoSoAPutImpl<PS, M, CMDT, ViewT> {
    using host_space = Kokkos::HostSpace;
    typedef Cabana::AoSoA<CMDT, host_space> Aosoa;
    AoSoAPutImpl(PS* ps, const Aosoa, adios2::IO &io, adios2::Engine &engine) {}
  };
  template <typename PS, std::size_t M, typename CMDT, typename ViewT, typename T, typename... Types>
  struct AoSoAPutImpl<PS, M, CMDT, ViewT, T, Types...> {
    using host_space = Kokkos::HostSpace;
    typedef Cabana::AoSoA<CMDT, host_space> Aosoa;

    AoSoAPutImpl(PS* ps, const Aosoa &src, adios2::IO &io, adios2::Engine &engine) {
      enclose(ps, src, io, engine);
      AoSoAPutImpl<PS, M+1, CMDT, ViewT, Types...>(ps, src, io, engine);
    }
    void enclose(PS* ps, const Aosoa &src, adios2::IO &io, adios2::Engine &engine) {
      // get basic type of T (if T is an array-type)
      using element_T = typename std::remove_all_extents<T>::type;
      // get slice
      auto sliceM = Cabana::slice<M>(src);
      element_T* sliceM_ptr = sliceM.data();
      int num_soa = sliceM.extent( 0 );
      int A_stride_0 = sliceM.stride( 0 );
      int total_size = num_soa*A_stride_0;
      // get variable name
      std::string varname = "type";
      varname = varname + std::to_string(M); // "typeM"
      // set up variable
      adios2::Dims shape{static_cast<size_t>(total_size)};
      adios2::Dims start{0};
      adios2::Dims count{static_cast<size_t>(total_size)};
      adios2::Variable<element_T> var = io.DefineVariable<element_T>(varname, shape, start, count);
      // copy slice
      engine.Put(var, sliceM_ptr);
    }
  };
  //High level Adios2::Put for AoSoA
  template <typename PS, typename... Types>
  struct AoSoAPut<PS, MemberTypes<Types...> > {
    using host_space = Kokkos::HostSpace;
    typedef Cabana::AoSoA<PS_DTBool<MemberTypes<Types...>>, host_space> Aosoa;
    typedef PS_DTBool<MemberTypes<Types...>> CM_DT;

    AoSoAPut(PS* ps, const Aosoa &src, adios2::IO &io, adios2::Engine &engine) {
      AoSoAPutImpl<PS, 0, CM_DT, typename PS::kkLidView, Types...>(ps, src, io, engine);
    }
  };

  // Templated AoSoA Read
  // TODO


  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::checkpointWrite(std::string path) {
    const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("CabM checkpointWrite");
    Kokkos::Timer overall_timer;

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (!comm_rank) {
      fprintf(stderr, "doing adios2 stuff in cabm checkpointWrite\n");

      // TODO: Fix hardcoding of config location
      std::string config = "/gpfs/u/home/MPFS/MPFSmttw/scratch/pumipicAdios2/build-dcsRhel8-gcc74-pumipic/adios2.xml";
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
      adios2::Dims shape_elms{static_cast<size_t>(num_elems)};
      adios2::Dims start_elms{0};
      adios2::Dims count_elms{static_cast<size_t>(num_elems)};
      adios2::Variable<lid_t> var_ppe = io.DefineVariable<lid_t>("particles_per_element", shape_elms, start_elms, count_elms);
      adios2::Variable<gid_t> var_gids = io.DefineVariable<gid_t>("element_gids", shape_elms, start_elms, count_elms);
      adios2::Dims shape_ptcls{static_cast<size_t>(num_ptcls)};
      adios2::Dims start_ptcls{0};
      adios2::Dims count_ptcls{static_cast<size_t>(num_ptcls)};
      adios2::Variable<lid_t> var_ptcl_elems = io.DefineVariable<lid_t>("particle_elements", shape_ptcls, start_ptcls, count_ptcls);

      // go through AoSoA, constructing particles_per_element and particle_elements by counting number of active ptcls
      kkLidView particles_per_element_d("particles_per_element_d", num_elems);
      kkLidView particle_elements_d("particle_elements_d", num_ptcls);

      kkLidView particle_new_indices_d("particle_new_indices_d", capacity_);
      kkLidView particle_id_temp("particle_id_temp", 1);

      kkLidView parentElms_cpy = parentElms_;
      const auto soa_len = AoSoA_t::vector_length;
      const auto activeSliceIdx = aosoa_->number_of_members-1;
      auto active = Cabana::slice<activeSliceIdx>(*aosoa_);
      auto atomic = KOKKOS_LAMBDA(const lid_t& soa, const lid_t& tuple) {
        if (active.access(soa,tuple)) {
          lid_t elm = parentElms_cpy(soa);
          // count particles in each element
          Kokkos::atomic_increment<lid_t>(&particles_per_element_d(elm));
          // get particle element for each id in later MTVs
          int ptcl_id = Kokkos::atomic_fetch_add(&(particle_id_temp(0)),1);
          particle_elements_d(ptcl_id) = elm;
        }
      };
      Cabana::SimdPolicy<soa_len,execution_space> simd_policy(0, capacity_);
      Cabana::simd_parallel_for(simd_policy, atomic, "atomic");

      // Copy to host for non-BP4 compliance
      kkLidHostMirror particles_per_element_h = deviceToHost(particles_per_element_d);
      kkLidHostMirror particle_elements_h = deviceToHost(particle_elements_d);
      kkGidHostMirror element_to_gid_h = deviceToHost(element_to_gid);

      // AoSoA variable saving
      // copy AoSoA to host for non-BP4 compliance
      using HstAoSoA_t = Cabana::AoSoA<CM_DT,host_space>;
      HstAoSoA_t aosoa_h;
      aosoa_h.resize(capacity_);
      Cabana::deep_copy(aosoa_h, *aosoa_);
      
      engine.BeginStep();
      engine.Put(var_num_elems, num_elems);
      engine.Put(var_num_ptcls, num_ptcls);
      engine.Put(var_padding_start, padding_start);
      engine.Put(var_extra_padding, extra_padding);
      engine.Put(var_name, name);

      engine.Put(var_ppe, particles_per_element_h.data());
      engine.Put(var_ptcl_elems, particle_elements_h.data());
      engine.Put(var_gids, element_to_gid_h.data());

      AoSoAPut<CabM<DataTypes, MemSpace>, DataTypes>(this, aosoa_h, io, engine);
      engine.EndStep();

      engine.Close();
    }

    RecordTime("CabM checkpointWrite", overall_timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::checkpointRead(std::string path) {
    const auto btime = prebarrier();
    Kokkos::Profiling::pushRegion("CabM checkpointRead");
    Kokkos::Timer overall_timer;

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (!comm_rank) {
      fprintf(stderr, "doing adios2 stuff in cabm checkpointRead\n");

      // TODO: Fix hardcoding of config location
      std::string config = "/gpfs/u/home/MPFS/MPFSmttw/scratch/pumipicAdios2/build-dcsRhel8-gcc74-pumipic/adios2.xml";
      adios2::ADIOS adios(config, MPI_COMM_WORLD);
      adios2::IO io = adios.DeclareIO("readerIO");

      adios2::Engine engine = io.Open(path, adios2::Mode::Read);

      // single-element variables
      adios2::Variable<lid_t> var_num_elems = io.InquireVariable<lid_t>("num_elems");
      adios2::Variable<lid_t> var_num_ptcls = io.InquireVariable<lid_t>("num_ptcls");
      adios2::Variable<lid_t> var_padding_start = io.InquireVariable<lid_t>("padding_start");
      adios2::Variable<double> var_extra_padding = io.InquireVariable<double>("extra_padding");
      adios2::Variable<std::string> var_name = io.InquireVariable<std::string>("name");

      engine.BeginStep();
      engine.Get(var_num_elems, num_elems);
      engine.Get(var_num_ptcls, num_ptcls);
      engine.Get(var_padding_start, padding_start);
      engine.Get(var_extra_padding, extra_padding);
      engine.Get(var_name, name);
      engine.EndStep();

      // array-sized variables
      adios2::Variable<lid_t> var_ppe = io.InquireVariable<lid_t>("particles_per_element");
      adios2::Variable<lid_t> var_ptcl_elems = io.InquireVariable<lid_t>("particle_elements");
      adios2::Variable<gid_t> var_gids = io.InquireVariable<gid_t>("element_gids");
      Kokkos::View<lid_t*,host_space> particle_elements_h(Kokkos::ViewAllocateWithoutInitializing("particle_elements_h"), num_ptcls);
      Kokkos::View<lid_t*,host_space> particles_per_element_h(Kokkos::ViewAllocateWithoutInitializing("particles_per_element_h"), num_elems);
      Kokkos::View<gid_t*,host_space> element_gids_h(Kokkos::ViewAllocateWithoutInitializing("element_gids_h"), num_elems);
      
      // TODO: AoSoA variable

      // TODO: Check if types are the same before moving on

      engine.BeginStep();
      engine.Get(var_ptcl_elems, particle_elements_h.data());
      engine.Get(var_ppe, particles_per_element_h.data());
      engine.Get(var_gids, element_gids_h.data());
      engine.EndStep();

      kkLidView particle_elements_d(Kokkos::ViewAllocateWithoutInitializing("particle_elements_d"), num_ptcls);
      kkLidView particles_per_element_d(Kokkos::ViewAllocateWithoutInitializing("particles_per_element_d"), num_elems);
      kkGidView element_gids(Kokkos::ViewAllocateWithoutInitializing("element_gids"), num_elems);
      hostToDevice(particle_elements_d, particle_elements_h.data());
      hostToDevice(particles_per_element_d, particles_per_element_h.data());
      hostToDevice(element_gids, element_gids_h.data());

      engine.Close();

      // TODO: Use variables to create new AoSoA
    }

    RecordTime("CabM checkpointRead", overall_timer.seconds(), btime);
    Kokkos::Profiling::popRegion();
  }

}
