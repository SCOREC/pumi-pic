#pragma once
#include <Cabana_Core.hpp>

namespace pumipic {
  /* Appends a type T to a pp::MemberType and provides it as a cabana::MemberType
     Usage: typename AppendMT<Type, MemberTypes>::type */
  template <typename T, typename... Types> struct AppendMT;
  //High level copy from MTVs to AoSoA
  template <typename PS, typename... Types> struct CopyMTVsToAoSoA;
  //High level copy from AoSoA to MTVs
  template <typename PS, typename... Types> struct CopyParticlesToSendFromAoSoA;
  //Per type copy from MTVs to AoSoA
  template <typename Device, std::size_t M, typename CMDT, typename ViewT, typename... Types>
  struct CopyMTVsToAoSoAImpl;
  //Per type copy from AoSoA to MTVs
  template <typename PS, std::size_t M, typename CMDT, typename ViewT, typename... Types>
  struct CopyParticlesToSendFromAoSoAImpl;


  /**************** AppendMT ***************/

  //Append type to the end
  template <typename T, typename... Types>
  struct AppendMT<T, particle_structs::MemberTypes<Types...> > {
    static constexpr int size = 1 + Cabana::MemberTypes<Types...>::size;
    using type = Cabana::MemberTypes<Types..., T>; //Put T before Types... to put at beginning
  };
  template <typename DataTypes> using PS_DTBool = typename AppendMT<bool,DataTypes>::type;

  template <typename ViewT, std::size_t Rank>
  using CheckRank = typename std::enable_if<(ViewT::rank == Rank), void>::type;


  /**************** CopyMTVsToAoSoA ***************/

  //Rank 0 ViewToSoA
  template <std::size_t M, typename SoA_t, typename View_t>
  PP_INLINE CheckRank<View_t, 1> copyViewToSoa(SoA_t &dst, lid_t dstind,
                                               View_t src, lid_t srcind) {
    Cabana::get<M>(dst, dstind) = src(srcind);
  }
  //Rank 1 ViewToSoA
  template <std::size_t M, typename SoA_t, typename View_t>
  PP_INLINE CheckRank<View_t, 2> copyViewToSoa(SoA_t &dst, lid_t dstind,
                                               View_t src, lid_t srcind) {
    for (lid_t i = 0; i < src.extent(1); ++i)
      Cabana::get<M>(dst, dstind, i) = src(srcind, i);
  }
  //Rank 2 ViewToSoA
  template <std::size_t M, typename SoA_t, typename View_t>
  PP_INLINE CheckRank<View_t, 3> copyViewToSoa(SoA_t &dst, lid_t dstind,
                                               View_t src, lid_t srcind) {
    for (lid_t i = 0; i < src.extent(1); ++i)
      for (lid_t j = 0; j < src.extent(2); ++j)
        Cabana::get<M>(dst, dstind, i, j) = src(srcind, i, j);
  }
  //Rank 3 ViewToSoA
  template <std::size_t M, typename SoA_t, typename View_t>
  PP_INLINE CheckRank<View_t, 4> copyViewToSoa(SoA_t &dst, lid_t dstind,
                                               View_t src, lid_t srcind) {
    for (lid_t i = 0; i < src.extent(1); ++i)
      for (lid_t j = 0; j < src.extent(2); ++j)
        for (lid_t k = 0; k < src.extent(3); ++k)
          Cabana::get<M>(dst, dstind, i, j, k) = src(srcind, i, j, k);
  }

  //Per type copy from MTVs to AoSoA
  template <typename PS, std::size_t M, typename CMDT, typename ViewT>
  struct CopyMTVsToAoSoAImpl<PS, M, CMDT, ViewT> {
    typedef typename PS::device_type Device;
    typedef typename PS::memory_space memory_space;
    typedef Cabana::AoSoA<CMDT, memory_space> Aosoa;
    CopyMTVsToAoSoAImpl(Aosoa, MemberTypeViewsConst, ViewT, ViewT) {}
  };

  template <typename PS, std::size_t M, typename CMDT, typename ViewT,
            typename T, typename... Types>
  struct CopyMTVsToAoSoAImpl<PS, M, CMDT, ViewT, T, Types...> {
    typedef typename PS::device_type Device;
    typedef typename PS::memory_space memory_space;
    typedef Cabana::AoSoA<CMDT, memory_space> Aosoa;
    CopyMTVsToAoSoAImpl(Aosoa &dst, MemberTypeViewsConst srcs, ViewT soa_indices,
                        ViewT soa_ptcl_indices) {
      enclose(dst, srcs, soa_indices, soa_ptcl_indices);
      CopyMTVsToAoSoAImpl<PS, M+1, CMDT, ViewT, Types...>(dst, srcs+1,
                                                              soa_indices,
                                                              soa_ptcl_indices);
    }
    void enclose(Aosoa &dst, MemberTypeViewsConst srcs, ViewT soa_indices,
                 ViewT soa_ptcl_indices) {
      MemberTypeView<T, Device> src =
        *static_cast<MemberTypeView<T, Device> const*>(srcs[0]);
      auto MTVToAoSoA = KOKKOS_LAMBDA(const int index) {
        const lid_t soa = soa_indices(index);
        const lid_t pid = soa_ptcl_indices(index);
        copyViewToSoa<M>(dst.access(soa), pid, src, index);
      };
      Kokkos::parallel_for("copyMTVtoAoSoA", soa_indices.size(), MTVToAoSoA);
    }
  };

  //High level copy from MTVs to AoSoA
  template <typename PS, typename... Types>
  struct CopyMTVsToAoSoA<PS, MemberTypes<Types...>> {
    typedef typename PS::device_type Device;
    typedef typename PS::memory_space memory_space;
    typedef Cabana::AoSoA<PS_DTBool<MemberTypes<Types...>>, memory_space> Aosoa;
    typedef PS_DTBool<MemberTypes<Types...>> CM_DT;

    CopyMTVsToAoSoA(Aosoa &dst, MemberTypeViewsConst src,
                    typename PS::kkLidView soa_indices,
                    typename PS::kkLidView soa_ptcl_indices) {
      if (src != NULL)
        CopyMTVsToAoSoAImpl<PS, 0, CM_DT, typename PS::kkLidView,
                            Types...>(dst, src, soa_indices, soa_ptcl_indices);
    }
  };

  /**************** CopyParticlesToSendFromAoSoA ***************/

  //Rank 0
  template <std::size_t M, typename View_t, typename Slice_t>
  PP_INLINE CheckRank<View_t, 1> copySliceToView(View_t dst, lid_t dstind,
                                               Slice_t &src, lid_t srcind) {
    dst(dstind) = src(srcind);
  }
  //Rank 1
  template <std::size_t M, typename View_t, typename Slice_t>
  PP_INLINE CheckRank<View_t, 2> copySliceToView(View_t dst, lid_t dstind,
                                               Slice_t &src, lid_t srcind) {
    for (lid_t i = 0; i < dst.extent(1); ++i)
      dst(dstind, i) = src(srcind, i);
  }
  //Rank 2
  template <std::size_t M, typename View_t, typename Slice_t>
  PP_INLINE CheckRank<View_t, 3> copySliceToView(View_t dst, lid_t dstind,
                                               Slice_t &src, lid_t srcind) {
    for (lid_t i = 0; i < dst.extent(1); ++i)
      for (lid_t j = 0; j < dst.extent(2); ++j)
        dst(dstind, i, j) = src(srcind, i, j);
  }
  //Rank 3
  template <std::size_t M, typename View_t, typename Slice_t>
  PP_INLINE CheckRank<View_t, 4> copySliceToView(View_t dst, lid_t dstind,
                                               Slice_t &src, lid_t srcind) {
    for (lid_t i = 0; i < dst.extent(1); ++i)
      for (lid_t j = 0; j < dst.extent(2); ++j)
        for (lid_t k = 0; k < dst.extent(3); ++k)
          dst(dstind, i, j, k) = src(srcind, i, j, k);
  }

  //Per type copy from AoSoA to MTVs
  template <typename PS, std::size_t M, typename CMDT, typename ViewT>
  struct CopyParticlesToSendFromAoSoAImpl<PS, M, CMDT, ViewT> {
    typedef typename PS::device_type Device;
    typedef typename PS::memory_space memory_space;
    typedef Cabana::AoSoA<CMDT, memory_space> Aosoa;
    CopyParticlesToSendFromAoSoAImpl(PS* ps, MemberTypeViews, const Aosoa,
                            typename PS::kkLidView, typename PS::kkLidView) {}
  };

  template <typename PS, std::size_t M, typename CMDT, typename ViewT, typename T, typename... Types>
  struct CopyParticlesToSendFromAoSoAImpl<PS, M, CMDT, ViewT, T, Types...> {
    typedef typename PS::device_type Device;
    typedef typename PS::memory_space memory_space;
    typedef Cabana::AoSoA<CMDT, memory_space> Aosoa;

    CopyParticlesToSendFromAoSoAImpl(PS* ps, MemberTypeViews dsts, const Aosoa &src,
                            typename PS::kkLidView ps_to_array,
                            typename PS::kkLidView array_indices) {
      enclose(ps, dsts, src, ps_to_array, array_indices);
      CopyParticlesToSendFromAoSoAImpl<PS, M+1, CMDT, ViewT,
                                  Types...>(ps, dsts+1, src, ps_to_array, array_indices);
    }
    void enclose(PS* ps, MemberTypeViews dsts, const Aosoa &src,
                 typename PS::kkLidView ps_to_array,
                 typename PS::kkLidView array_indices) {
      int comm_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
      MemberTypeView<T, Device> dst = *static_cast<MemberTypeView<T, Device>*>(dsts[0]);
      auto sliceM = Cabana::slice<M>(src);
      auto copyPSToArray = PS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const int arr_index = ps_to_array(ptcl_id);
        if (mask && arr_index != comm_rank) {
          const int index = array_indices(ptcl_id);
          copySliceToView<M>(dst, index, sliceM, ptcl_id);
        }
      };
      parallel_for(ps, copyPSToArray);
    }
  };

  //High level copy from AoSoA to MTVs
  template <typename PS, typename... Types>
  struct CopyParticlesToSendFromAoSoA<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    typedef typename PS::memory_space memory_space;
    typedef Cabana::AoSoA<PS_DTBool<MemberTypes<Types...>>, memory_space> Aosoa;
    typedef PS_DTBool<MemberTypes<Types...>> CM_DT;

    CopyParticlesToSendFromAoSoA(PS* ps, MemberTypeViews dsts, const Aosoa &src,
                        typename PS::kkLidView ps_to_array,
                        typename PS::kkLidView array_indices) {
      CopyParticlesToSendFromAoSoAImpl<PS, 0, CM_DT, typename PS::kkLidView,
                            Types...>(ps, dsts, src, ps_to_array, array_indices);
    }
  };

}
